#!/usr/bin/env python3
"""
LPA* planner node for local planning on an occupancy grid.

This module provides incremental path planning through sequential gate waypoints
using the LPA* algorithm with dynamic map updates.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Sequence

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, Float32MultiArray

from greenhorn_nav.utils.grid_processing import inflate_obstacles, detect_changed_cells
from greenhorn_nav.utils.path_utils import smooth_path
from greenhorn_nav.utils.planner_components import (
    PlannerConfig,
    PlannerState,
    MapCoordinateTransformer,
    GateNavigator,
    SegmentPlanner,
    MapProcessor,
    MapIndex,
    WorldXY
)


class LPAPlannerNode(Node):
    """LPA* planner node for local planning on an occupancy grid."""
    
    def __init__(self):
        super().__init__('lpa_planner_node')
        
        # Initialize configuration and state
        self.config = self._load_config()
        self.state = PlannerState()
        
        # Initialize components
        self.transformer = MapCoordinateTransformer(self)
        self.map_processor = MapProcessor(self, self.config)
        self.gate_navigator = GateNavigator(self, self.config, self.transformer)
        self.segment_planner = SegmentPlanner(self, self.config, self.transformer)
        
        # Setup ROS interfaces
        self._setup_ros_interfaces()
        
        self.get_logger().info(
            "LPAPlannerNode initialized (sequential gate planning)"
        )
    
    def _load_config(self) -> PlannerConfig:
        """Load configuration from ROS parameters."""
        config = PlannerConfig()
        
        # Declare and read all parameters
        params = {
            'map_topic': config.map_topic,
            'gate_midpoints_topic': config.gate_midpoints_topic,
            'boat_pose_topic': config.boat_pose_topic,
            'local_path_topic': config.local_path_topic,
            'frame_id': config.frame_id,
            'inflation_radius': config.inflation_radius,
            'cycle_hz': config.cycle_hz,
            'planner_heuristic': config.planner_heuristic,
            'smoothing_enabled': config.smoothing_enabled,
            'smoothing_factor': config.smoothing_factor,
            'smoothing_min_points': config.smoothing_min_points,
            'smoothing_max_points': config.smoothing_max_points,
            'smoothing_multiplier': config.smoothing_multiplier,
            'max_extraction_iters': config.max_extraction_iters,
            'qos_depth': config.qos_depth,
            'planning_horizon_distance': config.planning_horizon_distance,
            'min_gates_ahead': config.min_gates_ahead,
            'max_gates_ahead': config.max_gates_ahead,
            'blocked_gate_retry_limit': config.blocked_gate_retry_limit,
            'allow_gate_skipping': config.allow_gate_skipping,
            'gate_passed_proximity_threshold': config.gate_passed_proximity_threshold,
            'cell_occupied_threshold': config.cell_occupied_threshold,
        }
        
        for name, default in params.items():
            self.declare_parameter(name, default)
            setattr(config, name, self.get_parameter(name).value)
        
        # Validate cycle_hz
        if config.cycle_hz <= 0:
            self.get_logger().warn("cycle_hz <= 0; forcing to 1.0 Hz")
            config.cycle_hz = 1.0
        
        return config
    
    def _setup_ros_interfaces(self):
        """Setup ROS publishers, subscribers, and timers."""
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.config.qos_depth
        )
        
        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, self.config.map_topic, self._map_callback, qos
        )
        self.midpoints_sub = self.create_subscription(
            Float32MultiArray, self.config.gate_midpoints_topic,
            self._midpoints_callback, qos
        )
        self.boat_sub = self.create_subscription(
            Float32MultiArray, self.config.boat_pose_topic,
            self._boat_callback, qos
        )
        
        # Publishers
        self.path_pub = self.create_publisher(
            Path, self.config.local_path_topic, qos
        )
        self.inflated_map_pub = self.create_publisher(
            OccupancyGrid, '/inflated_map', qos
        )
        
        # Timer
        self._timer = self.create_timer(
            1.0 / self.config.cycle_hz, self._planning_cycle
        )
    
    # ==================== Callbacks ====================
    
    def _map_callback(self, msg: OccupancyGrid):
        """Handle incoming occupancy grid map."""
        if self.map_processor.process_map(msg, self.state):
            self.transformer.update_map(self.state.map_msg, self.state.inflated)
            self._publish_inflated_map()
    
    def _midpoints_callback(self, msg: Float32MultiArray):
        """Handle incoming gate midpoints."""
        try:
            data = msg.data
            if len(data) % 3 != 0:
                self.get_logger().warn(
                    f"Gate midpoints data length {len(data)} not divisible by 3"
                )
                return
            
            new_midpoints = []
            for i in range(0, len(data), 3):
                x, y, heading = data[i], data[i+1], data[i+2]
                new_midpoints.append((float(x), float(y)))
            
            if new_midpoints != self.state.gate_midpoints:
                self._handle_gate_update(new_midpoints)
                
        except Exception as ex:
            self.get_logger().error(f"midpoints_cb failed: {ex}")
    
    def _handle_gate_update(self, new_midpoints: List[WorldXY]):
        """Handle changes to gate midpoints."""
        old_count = len(self.state.gate_midpoints)
        self.state.gate_midpoints = new_midpoints
        
        if len(new_midpoints) < old_count:
            self.get_logger().info(
                f"Gate count decreased from {old_count} to {len(new_midpoints)} "
                f"- resetting planner state"
            )
            self.state.current_gate_idx = 0
            self.state.passed_gates.clear()
            self.state.skipped_gates.clear()
            self.state.gate_blocked_count.clear()
        elif self.state.current_gate_idx >= len(self.state.gate_midpoints):
            self.get_logger().warn(
                f"Current gate index {self.state.current_gate_idx} out of bounds, "
                f"clamping to {len(self.state.gate_midpoints) - 1}"
            )
            self.state.current_gate_idx = max(0, len(self.state.gate_midpoints) - 1)
        
        self.state.lpa_instances = [None] * len(self.state.gate_midpoints)
    
    def _boat_callback(self, msg: Float32MultiArray):
        """Handle incoming boat pose."""
        try:
            if len(msg.data) >= 2:
                self.state.boat_pose = (float(msg.data[0]), float(msg.data[1]))
                self.get_logger().debug(
                    f"Boat pose: x={self.state.boat_pose[0]:.3f}, "
                    f"y={self.state.boat_pose[1]:.3f}"
                )
        except Exception as ex:
            self.get_logger().error(f"boat_cb failed: {ex}")
    
    # ==================== Planning Loop ====================
    
    def _planning_cycle(self):
        """Main periodic planning loop."""
        if not self._can_plan():
            return
        
        start = self.transformer.world_to_map(self.state.boat_pose)
        if start is None:
            self.get_logger().debug("start out of map bounds")
            return
        start = (int(start[0]), int(start[1]))
        
        # Update navigation state
        self.gate_navigator.update_passed_gates(self.state)
        self.gate_navigator.advance_current_gate(self.state)
        
        # Compute planning horizon
        end_gate_idx = self.gate_navigator.compute_planning_horizon(self.state, start)
        
        # Plan through gates
        full_path = self._plan_through_gates(start, end_gate_idx)
        
        if not full_path:
            self.get_logger().warn("No path found through gates")
            return
        
        # Smooth and publish path
        smoothed_path = self._smooth_path(full_path)
        self._publish_path(smoothed_path)
    
    def _can_plan(self) -> bool:
        """Check if we have all necessary data for planning."""
        return (self.state.inflated is not None and 
                self.state.boat_pose is not None and 
                bool(self.state.gate_midpoints))
    
    def _plan_through_gates(self, start: MapIndex, end_gate_idx: int) -> List[WorldXY]:
        """Plan path through multiple gate waypoints."""
        full_path = []
        current_start = start
        
        for gate_idx in range(self.state.current_gate_idx, end_gate_idx + 1):
            if gate_idx in self.state.skipped_gates or gate_idx in self.state.passed_gates:
                self.get_logger().debug(f"Skipping gate {gate_idx} (blocked or passed)")
                continue
            
            goal_world = self.state.gate_midpoints[gate_idx]
            goal = self.transformer.world_to_map(goal_world)
            
            if goal is None:
                self.get_logger().debug(f"Gate {gate_idx} out of map bounds")
                break
            goal = (int(goal[0]), int(goal[1]))
            
            segment_path = self.segment_planner.plan_segment(
                current_start, goal, gate_idx, self.state
            )
            
            if not segment_path:
                if not self._handle_blocked_gate(gate_idx):
                    break
                continue
            
            # Reset blocked count on success
            if gate_idx in self.state.gate_blocked_count:
                self.state.gate_blocked_count[gate_idx] = 0
            
            # Add to full path (avoid duplicates at segment boundaries)
            if full_path and segment_path:
                segment_path = segment_path[1:]
            full_path.extend(segment_path)
            
            current_start = goal
        
        return full_path
    
    def _handle_blocked_gate(self, gate_idx: int) -> bool:
        """Handle a blocked gate. Returns True if planning should continue."""
        self.get_logger().warn(f"No path found to gate {gate_idx}")
        
        self.state.gate_blocked_count[gate_idx] = \
            self.state.gate_blocked_count.get(gate_idx, 0) + 1
        
        if self.state.gate_blocked_count[gate_idx] >= self.config.blocked_gate_retry_limit:
            self.get_logger().warn(
                f"Gate {gate_idx} blocked {self.state.gate_blocked_count[gate_idx]} "
                f"times - marking as skipped"
            )
            self.state.skipped_gates.add(gate_idx)
            
            if gate_idx == self.state.current_gate_idx:
                self.get_logger().info(f"Advancing past blocked gate {gate_idx}")
                self.state.current_gate_idx = min(
                    gate_idx + 1, len(self.state.gate_midpoints) - 1
                )
                return True
        
        return False
    
    # ==================== Path Processing ====================
    
    def _smooth_path(self, path: List[WorldXY]) -> List[WorldXY]:
        """Apply path smoothing if enabled."""
        if not self.config.smoothing_enabled:
            return path
        
        try:
            n_raw = len(path)
            num_points = max(
                self.config.smoothing_min_points,
                min(self.config.smoothing_max_points, 
                    int(n_raw * self.config.smoothing_multiplier))
            )
            
            smooth = smooth_path(
                path,
                smooth_factor=self.config.smoothing_factor,
                num_points=num_points
            )
            return [tuple(p) for p in np.asarray(smooth)]
            
        except Exception as ex:
            self.get_logger().warn(f"smoothing failed: {ex}")
            return path
    
    # ==================== Publishers ====================
    
    def _publish_path(self, waypoints: Sequence[WorldXY]):
        """Publish path as ROS Path message."""
        if not waypoints:
            return
        
        msg = Path()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.config.frame_id
        
        for x, y in waypoints:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        
        self.path_pub.publish(msg)
    
    def _publish_inflated_map(self):
        """Publish the inflated occupancy grid for visualization."""
        if self.state.map_msg is None or self.state.inflated is None:
            return
        
        msg = OccupancyGrid()
        msg.header = self.state.map_msg.header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = self.state.map_msg.info
        
        inflated_occup = (self.state.inflated * 100).astype(np.int8)
        msg.data = inflated_occup.flatten('C').tolist()
        
        self.inflated_map_pub.publish(msg)


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