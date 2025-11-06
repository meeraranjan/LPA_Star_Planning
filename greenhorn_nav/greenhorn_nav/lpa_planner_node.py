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
        self.declare_parameter('local_goal_topic', '/local_goal')
        self.declare_parameter('boat_pose_topic', '/boat_pose')
        self.declare_parameter('local_path_topic', '/local_path')
        self.declare_parameter('frame_id', 'map')

        self.declare_parameter('inflation_radius', 0.1)     # meters
        self.declare_parameter('cycle_hz', 2.0)
        self.declare_parameter('planner_heuristic', 'euclidean')
        self.declare_parameter('smoothing_enabled', True)  # or whatever default
        self.declare_parameter('smoothing_factor', 0.5)
        self.declare_parameter('smoothing_min_points', 10)
        self.declare_parameter('smoothing_max_points', 100)
        self.declare_parameter('max_extraction_iters', 20000)
        self.declare_parameter('qos_depth', 10)

        # QoS and topics
        qos_depth = int(self.get_parameter('qos_depth').value)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth
        )

        self.map_topic = self.get_parameter('map_topic').value
        self.goal_topic = self.get_parameter('local_goal_topic').value
        self.boat_topic = self.get_parameter('boat_pose_topic').value
        self.path_topic = self.get_parameter('local_path_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        
        self.smoothing_enabled = self.get_parameter('smoothing_enabled').value
        self.smoothing_factor = self.get_parameter('smoothing_factor').value
        self.smoothing_min_points = self.get_parameter('smoothing_min_points').value
        self.smoothing_max_points = self.get_parameter('smoothing_max_points').value
        

        # Subscriptions & publishers
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_cb, qos)
        self.goal_sub = self.create_subscription(PoseStamped, self.goal_topic, self.goal_cb, qos)
        self.boat_sub = self.create_subscription(Float32MultiArray, self.boat_topic, self.boat_cb, qos)
        self.path_pub = self.create_publisher(Path, self.path_topic, qos)
        self.inflated_map_pub = self.create_publisher(OccupancyGrid, '/inflated_map', qos)

        # State
        self.map_msg: Optional[OccupancyGrid] = None
        self.inflated: Optional[np.ndarray] = None
        self.prev_inflated: Optional[np.ndarray] = None
        self.goal: Optional[WorldXY] = None
        self.boat_pose: Optional[WorldXY] = None
        self.lpa: Optional[LPAStar] = None
        self.last_start: Optional[MapIndex] = None

        # Timer according to cycle_hz
        cycle_hz = float(self.get_parameter('cycle_hz').value)
        if cycle_hz <= 0:
            self.get_logger().warn("cycle_hz <= 0; forcing to 1.0 Hz")
            cycle_hz = 1.0
        self._timer = self.create_timer(1.0 / cycle_hz, self.cycle)

        self.get_logger().info("LPAPlannerNode initialized")

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
        
        # Convert binary inflated map (0/1) to occupancy values (0/100)
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
            binmap = (arr > 50).astype(np.uint8)
            if self.inflated is not None:
                self.prev_inflated = self.inflated.copy()

            # Inflate current map
            inflation_radius = float(self.get_parameter('inflation_radius').value)
            self.inflated = inflate_obstacles(binmap, inflation_radius=inflation_radius, resolution=res)

            # Publish inflated map
            self._publish_inflated_map()


            # self.get_logger().debug(f"Map received: {width}x{height} res={res} inflation={inflation_radius}")
        except Exception as ex:
            self.get_logger().error(f"map_cb failed: {ex}")

    def goal_cb(self, msg: PoseStamped):
        """Receive a local goal in world coordinates.

        Args:
            msg (PoseStamped): PoseStamped message with goal coordinates.
        """
        try:
            self.goal = (float(msg.pose.position.x), float(msg.pose.position.y))
            self.get_logger().info(f"New goal: x={self.goal[0]:.3f}, y={self.goal[1]:.3f}")
        except Exception as ex:
            self.get_logger().error(f"goal_cb failed: {ex}")

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

        Computes LPA* path from current boat position to goal,
        handles map changes, start/goal updates, path extraction, and smoothing.
        """
        if self.inflated is None or self.boat_pose is None or self.goal is None:
            return
        
        start = self.world_to_map(self.boat_pose)
        goal = self.world_to_map(self.goal)
        if start is None or goal is None:
            self.get_logger().debug("start or goal out of map bounds")
            return

        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))

        # Initialize or (re)create LPA* instance as required
        if self.lpa is None:
            self.get_logger().info("Creating new LPA* instance")
            if not self._create_lpa(start, goal):
                return

        # Handle moving goal (recreate LPA* from scratch)
        if self.lpa.goal != goal:
            self.get_logger().info(f"Goal changed from {self.lpa.goal} to {goal} -> recreating planner")
            if not self._create_lpa(start, goal):
                return

        # Handle map changes BEFORE updating start position
        if self.prev_inflated is not None:
            changed = detect_changed_cells(self.prev_inflated, self.inflated)
            if changed:
                self.get_logger().info(f"*** MAP CHANGED: {len(changed)} cells affected ***")

                # make sure planner sees the new grid array object
                if self.lpa is not None:
                    self.lpa.grid = self.inflated

                for cell in changed:
                    old_val = self.prev_inflated[cell]
                    new_val = self.inflated[cell]
                    self.get_logger().info(f"  Cell {cell}: {old_val} -> {new_val} (obstacle={'added' if new_val > old_val else 'removed'})")
                    try:
                        # update the changed cell
                        self.lpa.update_vertex(cell)
                        # and also update immediate neighbors since their rhs may be affected
                        for nbr in self.lpa.neighbors(cell):
                            self.lpa.update_vertex(nbr)
                    except Exception as ex:
                        self.get_logger().error(f"update_vertex failed for {cell}: {ex}")

        # Handle moving robot (start)
        if start != self.lpa.start:
            if self.last_start is not None:
                delta_h = self.lpa.heuristic(self.last_start, start)
                self.lpa.km += delta_h
                self.get_logger().debug(f"Start moved from {self.last_start} to {start}, km += {delta_h:.3f}")
            self.last_start = start
            self.lpa.start = start

        # Run LPA*
        self.get_logger().info(f"Running compute_shortest_path (start={start}, goal={goal})")
        try:
            self.lpa.compute_shortest_path()
        except Exception as ex:
            self.get_logger().error(f"compute_shortest_path failed: {ex}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            return

        # Check reachability
        start_g = self.lpa.g.get(start, np.inf)
        if start_g == np.inf:
            self.get_logger().warn(f"No path: start has infinite g-value")
            return

        self.get_logger().info(f"Path found! Start g-value: {start_g:.2f}")

        # Extract path
        raw_path = self._extract_path(start, goal)
        if not raw_path:
            self.get_logger().warn("No path extracted despite finite g-value")
            return

        # Smoothing
        try:
            if bool(self.get_parameter('smoothing_enabled').value):
                n_raw = len(raw_path)
                min_p = int(self.get_parameter('smoothing_min_points').value)
                max_p = int(self.get_parameter('smoothing_max_points').value)
                num_points = max(min_p, min(max_p, int(n_raw * 2)))
                smooth_factor = float(self.get_parameter('smoothing_factor').value)
                smooth = smooth_path(raw_path, smooth_factor=smooth_factor, num_points=num_points)
                smooth = [tuple(p) for p in np.asarray(smooth)]
            else:
                smooth = raw_path
        except Exception as ex:
            self.get_logger().warn(f"smoothing failed: {ex}")
            smooth = raw_path

        self.get_logger().info(f"Publishing path with {len(smooth)} waypoints")
        self._publish_path(smooth)
    # -------------------
    # helper methods
    # -------------------
    def _create_lpa(self, start: MapIndex, goal: MapIndex) -> bool:
        """Create and initialize LPA* instance.

        Args:
            start (MapIndex): Start cell indices (i, j).
            goal (MapIndex): Goal cell indices (i, j).

        Returns:
            bool: True if LPA* initialized successfully, False otherwise.
        """
        try:
            heuristic = str(self.get_parameter('planner_heuristic').value)
            self.lpa = LPAStar(start=start, goal=goal, grid=self.inflated, heuristic=heuristic)
            self.lpa.km = 0.0
            self.lpa.initialize()
            self.last_start = start
            return True
        except Exception as ex:
            self.get_logger().error(f"Failed to create LPA*: {ex}")
            self.lpa = None
            return False

    def _extract_path(self, start: MapIndex, goal: MapIndex) -> List[WorldXY]:
        """Extract a path from start->goal using LPA*'s g-values.

        Args:
            start (MapIndex): Start cell indices (i, j).
            goal (MapIndex): Goal cell indices (i, j).

        Returns:
            List[WorldXY]: List of world coordinates along the path.
        """
        if self.lpa.g.get(start, np.inf) == np.inf:
            return []

        path: List[MapIndex] = [start]
        cur = start
        max_iter = int(self.get_parameter('max_extraction_iters').value)
        it = 0
        while cur != goal and it < max_iter:
            it += 1
            nbrs = self.lpa.neighbors(cur)
            best = None
            best_val = np.inf
            for n in nbrs:
                g_n = self.lpa.g.get(n, np.inf)
                if g_n == np.inf:
                    continue
                val = self.lpa.cost(cur, n) + g_n
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
        """Publish a list of waypoints as a ROS Path message.

        Args:
            waypoints (Sequence[WorldXY]): List of (x, y) coordinates.
        """
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
