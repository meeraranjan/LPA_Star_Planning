# greenhorn_nav/local_goal_selector.py

import rclpy
from rclpy.node import Node
import numpy as np
from math import cos, sin, atan2
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from greenhorn_nav.utils.gate_state import GateManager
from greenhorn_nav.utils.gate_computation import GateBuilder


class LocalGoalSelector(Node):
    """ROS2 Node that selects local goals for navigating through buoy gates.
    
    Handles both paired gates (red-green) and unpaired buoys (with virtual midpoints).
    """
    
    def __init__(self):
        super().__init__('local_goal_selector')
        
        # Parameters
        self._declare_parameters()
        params = self._get_parameters()
        
        # Core components
        self.gate_manager = GateManager(waypoint_pass_tol=params['waypoint_pass_tol'])
        self.gate_builder = GateBuilder(
            unpaired_red_offset=params['unpaired_red_offset'],
            unpaired_green_offset=params['unpaired_green_offset'],
            max_gate_match_distance=params['max_gate_match_distance']
        )
        
        # State
        self.boat_pose = {'x': 0.0, 'y': 0.0, 'heading': 0.0}
        self.red_buoys = {}
        self.green_buoys = {}
        self.pose_received = False
        self.buoys_received = False
        
        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/local_goal', 10)
        self.midpoints_pub = self.create_publisher(Float32MultiArray, '/gate_midpoints', 10)
        
        # Subscribers
        self.create_subscription(Float32MultiArray, params['red_topic'], 
                                self._on_red_buoys, 10)
        self.create_subscription(Float32MultiArray, params['green_topic'], 
                                self._on_green_buoys, 10)
        self.create_subscription(Float32MultiArray, params['boat_topic'], 
                                self._on_boat_pose, 10)
        
        # Timer
        self.create_timer(1.0 / params['update_rate_hz'], self._update_and_publish)
        
        self.get_logger().info(
            f"LocalGoalSelector initialized (red_offset={params['unpaired_red_offset']}m, "
            f"green_offset={params['unpaired_green_offset']}m)"
        )
    
    def _declare_parameters(self):
        """Declare all ROS parameters."""
        self.declare_parameter('red_buoys_topic', '/red_buoys')
        self.declare_parameter('green_buoys_topic', '/green_buoys')
        self.declare_parameter('boat_pose_topic', '/boat_pose')
        self.declare_parameter('unpaired_red_offset', 0.5)
        self.declare_parameter('unpaired_green_offset', 0.5)
        self.declare_parameter('update_rate_hz', 2.0)
        self.declare_parameter('waypoint_pass_tol', 0.07)
        self.declare_parameter('max_gate_match_distance', 5.0)
    
    def _get_parameters(self) -> dict:
        """Get all parameter values."""
        return {
            'red_topic': self.get_parameter('red_buoys_topic').value,
            'green_topic': self.get_parameter('green_buoys_topic').value,
            'boat_topic': self.get_parameter('boat_pose_topic').value,
            'unpaired_red_offset': self.get_parameter('unpaired_red_offset').value,
            'unpaired_green_offset': self.get_parameter('unpaired_green_offset').value,
            'update_rate_hz': self.get_parameter('update_rate_hz').value,
            'waypoint_pass_tol': self.get_parameter('waypoint_pass_tol').value,
            'max_gate_match_distance': self.get_parameter('max_gate_match_distance').value,
        }
    
    def _on_red_buoys(self, msg: Float32MultiArray):
        """Parse red buoy locations [x0, y0, x1, y1, ...]."""
        self.red_buoys = self._parse_buoy_array(msg.data, 'r')
        self._recompute_gates()
    
    def _on_green_buoys(self, msg: Float32MultiArray):
        """Parse green buoy locations [x0, y0, x1, y1, ...]."""
        self.green_buoys = self._parse_buoy_array(msg.data, 'g')
        self._recompute_gates()
    
    def _on_boat_pose(self, msg: Float32MultiArray):
        """Update boat pose [x, y, heading]."""
        self.boat_pose['x'] = msg.data[0]
        self.boat_pose['y'] = msg.data[1]
        self.boat_pose['heading'] = msg.data[2]
        
        if not self.pose_received:
            self.pose_received = True
            self.get_logger().info(
                f"First pose: ({self.boat_pose['x']:.2f}, {self.boat_pose['y']:.2f})"
            )
    
    @staticmethod
    def _parse_buoy_array(data: list, prefix: str) -> dict:
        """Convert flat array [x0, y0, x1, y1, ...] to dict."""
        n = len(data) // 2
        return {f"{prefix}{i}": {"loc": [data[2*i], data[2*i + 1]]} for i in range(n)}
    
    def _recompute_gates(self):
        """Recompute gate sequence from current buoy data."""
        if not self.red_buoys and not self.green_buoys:
            return
        
        self.buoys_received = True
        
        # Filter out used buoys
        red_filtered, green_filtered = self.gate_manager.filter_used_buoys(
            self.red_buoys, self.green_buoys
        )
        
        if not red_filtered and not green_filtered:
            self.get_logger().info("All buoys used. No more gates.")
            return
        
        # Build new gate sequence
        boat_pos = np.array([self.boat_pose['x'], self.boat_pose['y']])
        gates = self.gate_builder.build_gates(
            red_filtered, green_filtered,
            boat_pos, self.gate_manager.passed_midpoints
        )
        
        self.gate_manager.update_gates(gates)
        
        # Log stats
        stats = self.gate_manager.stats
        paired = sum(1 for g in gates if not g.is_virtual)
        virtual = len(gates) - paired
        # self.get_logger().info(
        #     f"Computed {len(gates)} gates ({paired} paired, {virtual} virtual). "
        #     f"Used: {stats['used_red']} red, {stats['used_green']} green"
        # )
    
    def _update_and_publish(self):
        """Main update loop: check gate passage, compute goal, publish."""
        if not self.pose_received:
            self.get_logger().warn("Waiting for boat pose...", throttle_duration_sec=2.0)
            return
        
        if not self.gate_manager.gates:
            if not self.buoys_received:
                self.get_logger().warn("No buoy data...", throttle_duration_sec=2.0)
            return
        
        # Publish gate midpoints
        self._publish_midpoints()
        
        # Check if current gate is passed
        if self.gate_manager.check_and_advance(
            self.boat_pose['x'], self.boat_pose['y'], self.boat_pose['heading']
        ):
            gate = self.gate_manager.gates[self.gate_manager.current_idx - 1]
            gate_type = "virtual" if gate.is_virtual else "paired"
            self.get_logger().info(
                f"Gate passed ({gate_type}): red={gate.red_key}, green={gate.green_key}"
            )
            # Recompute to exclude used buoys
            self._recompute_gates()
        
        # Compute and publish goal
        goal_pos, goal_heading = self._compute_goal()
        self._publish_goal(goal_pos, goal_heading)
    
    def _publish_midpoints(self):
        """Publish all gate midpoints with headings."""
        msg = Float32MultiArray()
        msg.data = []
        for gate in self.gate_manager.gates:
            msg.data.extend([gate.midpoint[0], gate.midpoint[1], gate.heading])
        self.midpoints_pub.publish(msg)
    
    def _compute_goal(self) -> tuple:
        """Compute goal position and heading.
        
        Returns:
            (goal_position, goal_heading) as (np.ndarray, float)
        """
        gate = self.gate_manager.get_current_gate()
        
        if gate and not self.gate_manager.all_gates_passed:
            # Navigate toward current gate
            goal_pos = gate.midpoint
            
            # Heading: point toward gate (or use stored heading if at gate)
            vec_to_goal = goal_pos - np.array([self.boat_pose['x'], self.boat_pose['y']])
            if np.linalg.norm(vec_to_goal) > 0.001:
                heading = atan2(vec_to_goal[1], vec_to_goal[0])
            else:
                heading = gate.heading
        
        else:
            # All gates passed: lookahead along last heading
            last_gate = self.gate_manager.gates[-1]
            lookahead = 1.0
            goal_pos = np.array([self.boat_pose['x'], self.boat_pose['y']]) + \
                       lookahead * np.array([cos(last_gate.heading), sin(last_gate.heading)])
            heading = last_gate.heading
        
        return goal_pos, heading
    
    def _publish_goal(self, position: np.ndarray, heading: float):
        """Publish goal as PoseStamped."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.orientation.z = sin(heading / 2.0)
        msg.pose.orientation.w = cos(heading / 2.0)
        self.goal_pub.publish(msg)


def main(args=None):
    """Entry point for local_goal_selector node."""
    rclpy.init(args=args)
    node = LocalGoalSelector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()