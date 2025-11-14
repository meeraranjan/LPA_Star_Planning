#!/usr/bin/env python3

from __future__ import annotations

import math
import numpy as np
from typing import Optional, Tuple, List, Sequence

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, Float32MultiArray

from greenhorn_nav.utils.grid_processing import inflate_obstacles, detect_changed_cells
from greenhorn_nav.utils.lpa_helpers import LPAStar
from greenhorn_nav.utils.path_utils import smooth_path


MapIndex = Tuple[int, int]
WorldXY = Tuple[float, float]


class LPAPlannerNode(Node):
    """LPA* planner node for local planning on an occupancy grid.

    Subscribes to occupancy grid, local goal, and boat pose.
    Publishes a planned local path as a ROS Path message.
    """

    def __init__(self):
        """Initialize the LPAPlannerNode, parameters, publishers, subscribers, and timers."""
        super().__init__('lpa_planner_node')

        # -------------------
        # Parameters 
        # -------------------
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('gate_midpoints_topic', '/gate_midpoints')
        self.declare_parameter('boat_pose_topic', '/boat_pose')
        self.declare_parameter('local_path_topic', '/local_path')
        self.declare_parameter('frame_id', 'map')

        self.declare_parameter('inflation_radius', 0.1)     # meters
        self.declare_parameter('cycle_hz', 2.0)
        self.declare_parameter('planner_heuristic', 'euclidean')
        self.declare_parameter('smoothing_enabled', True)
        self.declare_parameter('smoothing_factor', 0.5)
        self.declare_parameter('smoothing_min_points', 10)
        self.declare_parameter('smoothing_max_points', 100)
        self.declare_parameter('max_extraction_iters', 20000)
        self.declare_parameter('qos_depth', 10)
        self.declare_parameter('planning_horizon_distance', 15.0)  # meters - plan this far ahead
        self.declare_parameter('min_gates_ahead', 1)  # Always plan at least this many gates
        self.declare_parameter('max_gates_ahead', 4)  # Never plan more than this many gates
        self.declare_parameter('blocked_gate_retry_limit', 3)  # How many cycles to retry before skipping
        self.declare_parameter('allow_gate_skipping', True)  # Allow skipping blocked gates
        
        self.declare_parameter('gate_passed_proximity_threshold', 0.5)  # meters
        self.declare_parameter('cell_occupied_threshold', 50)          # occupancy threshold
        self.declare_parameter('smoothing_multiplier', 2)         # factor for num_points
        # QoS and topics
        qos_depth = int(self.get_parameter('qos_depth').value)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth
        )

        self.map_topic = self.get_parameter('map_topic').value
        self.midpoints_topic = self.get_parameter('gate_midpoints_topic').value
        self.boat_topic = self.get_parameter('boat_pose_topic').value
        self.path_topic = self.get_parameter('local_path_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        
        self.smoothing_enabled = self.get_parameter('smoothing_enabled').value
        self.smoothing_factor = self.get_parameter('smoothing_factor').value
        self.smoothing_min_points = self.get_parameter('smoothing_min_points').value
        self.smoothing_max_points = self.get_parameter('smoothing_max_points').value
        self.planning_horizon_distance = self.get_parameter('planning_horizon_distance').value
        self.min_gates_ahead = self.get_parameter('min_gates_ahead').value
        self.max_gates_ahead = self.get_parameter('max_gates_ahead').value
        self.blocked_gate_retry_limit = self.get_parameter('blocked_gate_retry_limit').value
        self.allow_gate_skipping = self.get_parameter('allow_gate_skipping').value
        self.gate_passed_proximity_threshold = self.get_parameter('gate_passed_proximity_threshold').value
        self.cell_occupied_threshold = self.get_parameter('cell_occupied_threshold').value
        self.smoothing_multiplier = self.get_parameter('smoothing_multiplier').value
        
        # Subscriptions & publishers
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, qos)
        self.midpoints_sub = self.create_subscription(Float32MultiArray, self.midpoints_topic, self.midpoints_cb, qos)
        self.boat_sub = self.create_subscription(Float32MultiArray, self.boat_topic, self.boat_cb, qos)
        self.path_pub = self.create_publisher(Path, self.path_topic, qos)
        self.inflated_map_pub = self.create_publisher(OccupancyGrid, '/inflated_map', qos)

        # State
        self.map_msg: Optional[OccupancyGrid] = None
        self.inflated: Optional[np.ndarray] = None
        self.prev_inflated: Optional[np.ndarray] = None
        self.gate_midpoints: List[WorldXY] = []  # List of (x, y) gate midpoints
        self.boat_pose: Optional[WorldXY] = None
        self.lpa_instances: List[Optional[LPAStar]] = []  # One LPA* per gate segment
        self.last_start: Optional[MapIndex] = None
        self.current_gate_idx: int = 0  # Which gate we're currently heading to
        self.gate_blocked_count: dict = {}  # Track how many times each gate has been blocked
        self.skipped_gates: set = set()  # Gates we've decided to skip
        self.gate_blocked_count: dict = {}  # Track how many times each gate has been blocked
        self.last_start: Optional[MapIndex] = None
        self.current_gate_idx: int = 0  # Which gate we're currently heading to

        # Timer according to cycle_hz
        cycle_hz = float(self.get_parameter('cycle_hz').value)
        if cycle_hz <= 0:
            self.get_logger().warn("cycle_hz <= 0; forcing to 1.0 Hz")
            cycle_hz = 1.0
        self._timer = self.create_timer(1.0 / cycle_hz, self.cycle)

        self.get_logger().info("LPAPlannerNode initialized (sequential gate planning)")

    # -------------------
    # Callbacks
    # -------------------
    def _publish_inflated_map(self):
        """Publish the inflated occupancy grid."""
        if self.map_msg is None or self.inflated is None:
            return
        
        msg = OccupancyGrid()
        msg.header = self.map_msg.header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = self.map_msg.info
        
        inflated_occup = (self.inflated * 100).astype(np.int8)
        msg.data = inflated_occup.flatten('C').tolist()
        
        self.inflated_map_pub.publish(msg)
        
    def map_cb(self, msg: OccupancyGrid):
        """Receive occupancy grid, convert to binary and inflate obstacles.

        Args:
            msg (OccupancyGrid): Occupancy grid message from ROS.
        """
        try:
            self.map_msg = msg
            width = msg.info.width
            height = msg.info.height
            res = msg.info.resolution
            arr = np.array(msg.data, dtype=np.int8).reshape((height, width))
            binmap = (arr > self.cell_occupied_threshold).astype(np.uint8)
            if self.inflated is not None:
                self.prev_inflated = self.inflated.copy()

            inflation_radius = float(self.get_parameter('inflation_radius').value)
            new_inflated = inflate_obstacles(binmap, inflation_radius=inflation_radius, resolution=res)
            
            # Detect changes immediately in callback
            if self.prev_inflated is not None:
                changed = detect_changed_cells(self.prev_inflated, new_inflated)
                if changed:
                    self.get_logger().info(f"*** MAP CHANGED IN CALLBACK: {len(changed)} cells affected ***")
                    
                    # Update inflated map
                    self.inflated = new_inflated
                    
                    # Immediately update ALL existing LPA* instances
                    for gate_idx, lpa in enumerate(self.lpa_instances):
                        if lpa is not None:
                            lpa.grid = self.inflated
                            for cell in changed:
                                old_val = self.prev_inflated[cell]
                                new_val = self.inflated[cell]
                                try:
                                    lpa.update_vertex(cell)
                                    for nbr in lpa.neighbors(cell):
                                        lpa.update_vertex(nbr)
                                except Exception as ex:
                                    self.get_logger().error(f"update_vertex failed for gate {gate_idx}, cell {cell}: {ex}")
                            self.get_logger().info(f"  Updated LPA* instance for gate {gate_idx} in callback")
                else:
                    self.inflated = new_inflated
            else:
                self.inflated = new_inflated
            
            self._publish_inflated_map()

        except Exception as ex:
            self.get_logger().error(f"map_cb failed: {ex}")

    def midpoints_cb(self, msg: Float32MultiArray):
        """Receive gate midpoints as a flattened array [x0, y0, heading0, x1, y1, heading1, ...]"""
        try:
            data = msg.data
            if len(data) % 3 != 0:
                self.get_logger().warn(f"Gate midpoints data length {len(data)} not divisible by 3")
                return
            
            new_midpoints = []
            for i in range(0, len(data), 3):
                x, y, heading = data[i], data[i+1], data[i+2]
                new_midpoints.append((float(x), float(y)))
            
            # Only update if changed
            if new_midpoints != self.gate_midpoints:
                self.gate_midpoints = new_midpoints
                # Reset LPA instances when gates change
                self.lpa_instances = [None] * len(self.gate_midpoints)
                self.get_logger().info(f"Received {len(self.gate_midpoints)} gate midpoints")
                
        except Exception as ex:
            self.get_logger().error(f"midpoints_cb failed: {ex}")

    def boat_cb(self, msg: Float32MultiArray):
        """Receive the boat pose as [x, y].

        Args:
            msg (Float32MultiArray): ROS message containing boat coordinates.
        """
        try:
            if len(msg.data) >= 2:
                self.boat_pose = (float(msg.data[0]), float(msg.data[1]))
                self.get_logger().debug(f"Boat pose: x={self.boat_pose[0]:.3f}, y={self.boat_pose[1]:.3f}")
        except Exception as ex:
            self.get_logger().error(f"boat_cb failed: {ex}")

    # -------------------
    # coordinate transforms
    # -------------------
    def world_to_map(self, xy: WorldXY) -> Optional[MapIndex]:
        """Convert world coordinates (x,y) to grid indices (i,j).

        Args:
            xy (WorldXY): World coordinates (x, y).

        Returns:
            Optional[MapIndex]: Grid indices (i, j), or None if out-of-bounds or map missing.
        """
        if self.map_msg is None or self.inflated is None:
            return None
        ox = self.map_msg.info.origin.position.x
        oy = self.map_msg.info.origin.position.y
        res = self.map_msg.info.resolution
        j = int((xy[0] - ox) / res)
        i = int((xy[1] - oy) / res)
        if 0 <= i < self.inflated.shape[0] and 0 <= j < self.inflated.shape[1]:
            return (i, j)
        return None

    def map_to_world(self, ij: MapIndex) -> WorldXY:
        """Convert grid index (i,j) to world coordinates (x,y) at cell center.

        Args:
            ij (MapIndex): Grid indices (i, j).

        Returns:
            WorldXY: World coordinates (x, y).
        """
        assert self.map_msg is not None
        ox = self.map_msg.info.origin.position.x
        oy = self.map_msg.info.origin.position.y
        res = self.map_msg.info.resolution
        x = ox + ij[1] * res + res * 0.5
        y = oy + ij[0] * res + res * 0.5
        return (x, y)

    # -------------------
    # main loop
    # -------------------
    def cycle(self):
        """Main periodic planner loop.

        Computes LPA* path from current boat position to goal, through all existing gates sequentially,
        handles map changes, start/goal updates, path extraction, and smoothing.
        """
        if self.inflated is None or self.boat_pose is None or not self.gate_midpoints:
            return
        
        start = self.world_to_map(self.boat_pose)
        if start is None:
            self.get_logger().debug("start out of map bounds")
            return
        start = (int(start[0]), int(start[1]))

        # Determine which gate we're currently targeting
        self._update_current_gate_index()
        
        # Skip blocked gates if enabled
        if self.allow_gate_skipping:
            self._skip_blocked_gates()
        
        # Determine which gates to plan through based on distance horizon
        end_gate_idx = self._compute_planning_horizon(start)
        
        self.get_logger().debug(f"Planning from gate {self.current_gate_idx} to {end_gate_idx}")
        
        # Plan through each gate segment sequentially
        full_path = []
        current_start = start
        
        for gate_idx in range(self.current_gate_idx, end_gate_idx + 1):
            # Skip gates marked as blocked
            if gate_idx in self.skipped_gates:
                self.get_logger().info(f"Skipping blocked gate {gate_idx}")
                continue
                
            goal_world = self.gate_midpoints[gate_idx]
            goal = self.world_to_map(goal_world)
            
            if goal is None:
                self.get_logger().debug(f"Gate {gate_idx} out of map bounds")
                break
            goal = (int(goal[0]), int(goal[1]))
            
            # Get or create LPA* for this segment
            segment_path = self._plan_segment(current_start, goal, gate_idx)
            
            if not segment_path:
                self.get_logger().warn(f"No path found to gate {gate_idx}")
                
                # Track blocked gate
                self.gate_blocked_count[gate_idx] = self.gate_blocked_count.get(gate_idx, 0) + 1
                
                # If blocked too many times, mark for skipping
                if self.gate_blocked_count[gate_idx] >= self.blocked_gate_retry_limit:
                    self.get_logger().warn(
                        f"Gate {gate_idx} blocked {self.gate_blocked_count[gate_idx]} times - "
                        f"marking as skipped"
                    )
                    self.skipped_gates.add(gate_idx)
                    
                    # Try to advance to next gate
                    if gate_idx == self.current_gate_idx:
                        self.get_logger().info(f"Advancing to next gate due to blockage")
                        self.current_gate_idx = min(gate_idx + 1, len(self.gate_midpoints) - 1)
                
                break
            
            # Reset blocked count on success
            self.gate_blocked_count[gate_idx] = 0
            
            # Add to full path (avoid duplicate points at segment boundaries)
            if full_path and segment_path:
                segment_path = segment_path[1:]
            full_path.extend(segment_path)
            
            # Next segment starts where this one ended
            current_start = goal
        
        if not full_path:
            self.get_logger().warn("No path found through gates")
            return
        
        # Smooth the entire path
        try:
            if bool(self.get_parameter('smoothing_enabled').value):
                n_raw = len(full_path)
                min_p = int(self.get_parameter('smoothing_min_points').value)
                max_p = int(self.get_parameter('smoothing_max_points').value)
                num_points = max(min_p, min(max_p, int(n_raw * self.smoothing_multiplier)))
                smooth_factor = float(self.get_parameter('smoothing_factor').value)
                smooth = smooth_path(full_path, smooth_factor=smooth_factor, num_points=num_points)
                smooth = [tuple(p) for p in np.asarray(smooth)]
            else:
                smooth = full_path
        except Exception as ex:
            self.get_logger().warn(f"smoothing failed: {ex}")
            smooth = full_path

        self.get_logger().info(f"Publishing path through gates {self.current_gate_idx} to {end_gate_idx} with {len(smooth)} waypoints")
        self._publish_path(smooth)

    def _update_current_gate_index(self):
        """Update which gate we're currently heading toward based on proximity."""
        if not self.boat_pose or not self.gate_midpoints:
            return
        
        boat_pos = np.array(self.boat_pose)
        
        # Check if we're close to current gate, if so advance
        if self.current_gate_idx < len(self.gate_midpoints):
            current_gate_pos = np.array(self.gate_midpoints[self.current_gate_idx])
            dist = np.linalg.norm(boat_pos - current_gate_pos)
            
            # If close to current gate (within 0.5m), advance to next
            if dist < self.gate_passed_proximity_threshold and self.current_gate_idx < len(self.gate_midpoints) - 1:
                self.current_gate_idx += 1
                self.get_logger().info(f"Advanced to gate {self.current_gate_idx}")
                
    def _is_gate_blocked(self, gate_idx: int) -> bool:
        """Check if a gate midpoint is inside an obstacle."""
        if gate_idx >= len(self.gate_midpoints):
            return True
        goal_world = self.gate_midpoints[gate_idx]
        goal_map = self.world_to_map(goal_world)
        if goal_map is None:
            return True
        i, j = goal_map
        return self.inflated[i, j] > 0

    def _skip_blocked_gates(self):
        """Skip past any gates that are blocked."""
        while (self.current_gate_idx < len(self.gate_midpoints) and
            (self.current_gate_idx in self.skipped_gates or self._is_gate_blocked(self.current_gate_idx))):
            self.get_logger().info(f"Skipping blocked gate {self.current_gate_idx}")
            self.skipped_gates.add(self.current_gate_idx)
            self.current_gate_idx += 1

    def _compute_planning_horizon(self, start: MapIndex) -> int:
        """Compute how many gates ahead to plan based on distance.
        
        Args:
            start: Current boat position in map coordinates
            
        Returns:
            Index of the farthest gate to plan to
        """
        if not self.gate_midpoints:
            return 0
        
        boat_world = self.map_to_world(start)
        boat_pos = np.array(boat_world)
        
        # Start from current gate
        end_gate_idx = self.current_gate_idx
        cumulative_distance = 0.0
        
        # Keep adding gates until we exceed distance horizon or hit max gates
        for gate_idx in range(self.current_gate_idx, len(self.gate_midpoints)):
            gate_pos = np.array(self.gate_midpoints[gate_idx])
            
            # Distance from boat to this gate
            if gate_idx == self.current_gate_idx:
                dist_to_gate = np.linalg.norm(gate_pos - boat_pos)
            else:
                # Distance from previous gate to this gate
                prev_gate_pos = np.array(self.gate_midpoints[gate_idx - 1])
                dist_to_gate = np.linalg.norm(gate_pos - prev_gate_pos)
            
            cumulative_distance += dist_to_gate
            
            # Check if we've exceeded the distance horizon
            if cumulative_distance > self.planning_horizon_distance:
                break
            
            end_gate_idx = gate_idx
            
            # Check if we've hit max gates ahead
            if (gate_idx - self.current_gate_idx + 1) >= self.max_gates_ahead:
                break
        
        # Ensure we always plan at least min_gates_ahead
        min_end_idx = min(
            self.current_gate_idx + self.min_gates_ahead - 1,
            len(self.gate_midpoints) - 1
        )
        end_gate_idx = max(end_gate_idx, min_end_idx)
        
        self.get_logger().debug(
            f"Planning horizon: {cumulative_distance:.2f}m covers gates "
            f"{self.current_gate_idx} to {end_gate_idx}"
        )
        
        return end_gate_idx

    def _plan_segment(self, start: MapIndex, goal: MapIndex, gate_idx: int) -> List[WorldXY]:
        """Plan a path segment from start to a specific gate."""
        
        # Ensure we have enough LPA instances
        while len(self.lpa_instances) <= gate_idx:
            self.lpa_instances.append(None)
        
        lpa = self.lpa_instances[gate_idx]
        
        # Initialize or recreate LPA* if needed
        if lpa is None:
            self.get_logger().info(f"Creating new LPA* instance for gate {gate_idx}")
            if not self._create_lpa_for_gate(start, goal, gate_idx):
                return []
            lpa = self.lpa_instances[gate_idx]
        
        # Handle goal changes
        if lpa.goal != goal:
            self.get_logger().info(f"Gate {gate_idx} goal changed -> recreating planner")
            if not self._create_lpa_for_gate(start, goal, gate_idx):
                return []
            lpa = self.lpa_instances[gate_idx]
        
        # IMPORTANT: Always ensure grid reference is current
        lpa.grid = self.inflated
        
        # Handle moving start
        if start != lpa.start:
            if gate_idx == self.current_gate_idx:  # Only update km for current gate
                prev_start = lpa.start
                delta_h = lpa.heuristic(prev_start, start)
                lpa.km += delta_h
                self.get_logger().debug(f"Gate {gate_idx}: Start moved, km += {delta_h:.3f}")
            lpa.start = start
        
        # Run LPA*
        try:
            self.get_logger().debug(f"Running compute_shortest_path for gate {gate_idx}")
            lpa.compute_shortest_path()
        except Exception as ex:
            self.get_logger().error(f"compute_shortest_path failed for gate {gate_idx}: {ex}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return []
        
        # Check reachability
        start_g = lpa.g.get(start, np.inf)
        if start_g == np.inf:
            self.get_logger().warn(f"No path to gate {gate_idx}: start has infinite g-value")
            return []
        
        self.get_logger().debug(f"Gate {gate_idx}: Path found with start g-value: {start_g:.2f}")
        
        # Extract path
        return self._extract_path_from_lpa(start, goal, lpa)

    # -------------------
    # helper methods
    # -------------------
    def _create_lpa_for_gate(self, start: MapIndex, goal: MapIndex, gate_idx: int) -> bool:
        """Create and initialize LPA* instance for a specific gate."""
        try:
            heuristic = str(self.get_parameter('planner_heuristic').value)
            lpa = LPAStar(start=start, goal=goal, grid=self.inflated, heuristic=heuristic)
            lpa.km = 0.0
            lpa.initialize()
            self.lpa_instances[gate_idx] = lpa
            self.last_start = start
            return True
        except Exception as ex:
            self.get_logger().error(f"Failed to create LPA* for gate {gate_idx}: {ex}")
            self.lpa_instances[gate_idx] = None
            return False

    def _extract_path_from_lpa(self, start: MapIndex, goal: MapIndex, lpa: LPAStar) -> List[WorldXY]:
        """Extract a path from start->goal using LPA*'s g-values."""
        if lpa.g.get(start, np.inf) == np.inf:
            return []

        path: List[MapIndex] = [start]
        cur = start
        max_iter = int(self.get_parameter('max_extraction_iters').value)
        it = 0
        while cur != goal and it < max_iter:
            it += 1
            nbrs = lpa.neighbors(cur)
            best = None
            best_val = np.inf
            for n in nbrs:
                g_n = lpa.g.get(n, np.inf)
                if g_n == np.inf:
                    continue
                val = lpa.cost(cur, n) + g_n
                if val < best_val:
                    best_val = val
                    best = n
            if best is None:
                break
            path.append(best)
            cur = best

        if it >= max_iter:
            self.get_logger().warning("Reached max extraction iterations")

        # convert to world coordinates
        world_path = [self.map_to_world(p) for p in path]
        return world_path

    def _publish_path(self, waypoints: Sequence[WorldXY]):
        """Publish a list of waypoints as a ROS Path message."""
        if not waypoints:
            return
        msg = Path()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        for x, y in waypoints:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)


def main(args=None):
    """Main entry point for the LPAPlannerNode."""
    rclpy.init(args=args)
    node = LPAPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()