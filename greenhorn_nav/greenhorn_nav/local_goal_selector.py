# greenhorn_nav/local_goal_selector.py

import rclpy
from rclpy.node import Node
import numpy as np
from math import cos, sin, atan2
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from greenhorn_nav.utils.buoys2gates import buoys2gates, compute_gate_midpoint_and_heading


class LocalGoalSelector(Node):
    """ROS2 Node that selects local goals for the boat to navigate through gates.

    Handles both paired gates (red-green) and unpaired buoys (with virtual midpoints).
    The node subscribes to the boat's pose, publishes local goals, midpoints of gates,
    and tracks which gates have been passed.
    """
    def __init__(self):
        super().__init__('local_goal_selector')

        # Declare parameters
        self.declare_parameter('red_buoys_topic', '/red_buoys')
        self.declare_parameter('green_buoys_topic', '/green_buoys')
        self.declare_parameter('boat_pose_topic', '/boat_pose')
        self.declare_parameter('unpaired_red_offset', 0.5)
        self.declare_parameter('unpaired_green_offset', 0.5)
        self.declare_parameter('update_rate_hz', 2.0)
        self.declare_parameter('waypoint_pass_tol', 0.07)
        self.declare_parameter('max_gate_match_distance', 5.0)

        # Get parameters
        red_topic = self.get_parameter('red_buoys_topic').value
        green_topic = self.get_parameter('green_buoys_topic').value
        boat_topic = self.get_parameter('boat_pose_topic').value
        self.unpaired_red_offset = self.get_parameter('unpaired_red_offset').value
        self.unpaired_green_offset = self.get_parameter('unpaired_green_offset').value
        update_rate_hz = self.get_parameter('update_rate_hz').value
        self.waypoint_pass_tol = self.get_parameter('waypoint_pass_tol').value
        self.max_gate_match_distance= self.get_parameter('max_gate_match_distance').value
        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/local_goal', 10)
        self.midpoints_pub = self.create_publisher(Float32MultiArray, '/gate_midpoints', 10)
        
        # Timer
        self.timer = self.create_timer(1.0 / update_rate_hz, self.try_update_goal)

        # State
        self.current_pose = {'x': 0.0, 'y': 0.0, 'heading': 0.0}
        self.pose_received = False
        self.gate_sequence = []       # list of (midpoint, heading, is_virtual)
        self.gate_ids = []            # unique identifiers for gates
        self.current_gate_idx = 0

        # Track used buoys persistently - this is the key fix!
        self.used_red_buoys = set()   # Set of red buoy keys that have been used
        self.used_green_buoys = set() # Set of green buoy keys that have been used
        self.passed_gates = set()     # Set of gate IDs that have been passed
        self.passed_gate_midpoints = []  # List of midpoints from passed gates (for heading calculation)

        # Subscriptions
        self.red_dict = {}
        self.green_dict = {}
        self.buoys_received = False

        self.create_subscription(Float32MultiArray, red_topic, 
                                self.red_buoy_callback, 10)
        self.create_subscription(Float32MultiArray, green_topic, 
                                self.green_buoy_callback, 10)
        self.create_subscription(Float32MultiArray, boat_topic, 
                                self.pose_callback, 10)

        self.get_logger().info(
            f"LocalGoalSelector initialized with unpaired offsets: "
            f"red={self.unpaired_red_offset}m, green={self.unpaired_green_offset}m"
        )

    # ------------------ Callbacks ------------------
    def red_buoy_callback(self, msg):
        """Parse incoming red buoy locations [x0, y0, x1, y1, ...]"""
        data = msg.data
        n = len(data) // 2
        
        self.red_dict = {}
        for i in range(n):
            x = data[2*i]
            y = data[2*i + 1]
            self.red_dict[f"r{i}"] = {"loc": [x, y]}
        
        self._update_gates()

    def green_buoy_callback(self, msg):
        """Parse incoming green buoy locations [x0, y0, x1, y1, ...]"""
        data = msg.data
        n = len(data) // 2
        
        self.green_dict = {}
        for i in range(n):
            x = data[2*i]
            y = data[2*i + 1]
            self.green_dict[f"g{i}"] = {"loc": [x, y]}
        
        self._update_gates()

    def _filter_unused_buoys(self):
        """Remove already-used buoys from the dictionaries before gate computation."""
        # Filter out used red buoys
        filtered_red = {}
        for key, value in self.red_dict.items():
            if key not in self.used_red_buoys:
                filtered_red[key] = value
        
        # Filter out used green buoys
        filtered_green = {}
        for key, value in self.green_dict.items():
            if key not in self.used_green_buoys:
                filtered_green[key] = value
        
        return filtered_red, filtered_green

    def _update_gates(self):
        """Recompute gates when buoy data changes, excluding already-used buoys."""
        if not self.red_dict and not self.green_dict:
            return
        
        self.buoys_received = True
        
        # Filter out already-used buoys
        filtered_red, filtered_green = self._filter_unused_buoys()
        
        if not filtered_red and not filtered_green:
            self.get_logger().info("All buoys have been used. No more gates to compute.")
            return
        
        self.gate_sequence = []
        self.gate_ids = []
        boat_pos = np.array([self.current_pose['x'], self.current_pose['y']])
        
        # Compute gate matches using only unused buoys
        G, gate_matches, scores = buoys2gates(
            filtered_red, 
            filtered_green,
            max_match_distance=self.max_gate_match_distance,
            boat_position=boat_pos
        )

        # First pass: compute all midpoints (needed for heading calculation)
        gate_info = []  # List of (red_loc, green_loc, is_virtual, red_key, green_key)
        for match in gate_matches:
            red_key, green_key, is_virtual = match
            
            red_loc = np.array(G.nodes[red_key]['loc']) if red_key else None
            green_loc = np.array(G.nodes[green_key]['loc']) if green_key else None
            
            gate_info.append((red_loc, green_loc, is_virtual, red_key, green_key))

        # Compute preliminary midpoints for next-gate heading calculation
        # Include passed gate midpoints at the beginning for proper heading calculation
        all_preliminary_midpoints = self.passed_gate_midpoints.copy()
        
        for idx, (red_loc, green_loc, is_virtual, _, _) in enumerate(gate_info):
            # Index into all_preliminary_midpoints includes passed gates
            full_idx = len(self.passed_gate_midpoints) + idx
            
            if not is_virtual:
                # Paired gate: midpoint between red and green
                midpoint = (red_loc + green_loc) / 2.0
            elif red_loc is not None:
                # Unpaired red: need to compute offset midpoint
                prev_prelim = None
                if full_idx > 0:
                    prev_prelim = all_preliminary_midpoints[-1]
                
                # Estimate heading for offset
                if prev_prelim is not None:
                    vec_path = red_loc - prev_prelim
                    heading_est = np.arctan2(vec_path[1], vec_path[0])
                else:
                    heading_est = self.current_pose['heading']
                # Offset LEFT from heading
                left_angle = heading_est + np.pi / 2
                offset_vec = self.unpaired_red_offset * np.array([np.cos(left_angle), np.sin(left_angle)])
                midpoint = red_loc + offset_vec
            else:
                # Unpaired green: need to compute offset midpoint
                prev_prelim = None
                if full_idx > 0:
                    prev_prelim = all_preliminary_midpoints[-1]
                
                # Estimate heading for offset
                if prev_prelim is not None:
                    vec_path = green_loc - prev_prelim
                    heading_est = np.arctan2(vec_path[1], vec_path[0])
                else:
                    heading_est = self.current_pose['heading']
                # Offset RIGHT from heading
                right_angle = heading_est - np.pi / 2
                offset_vec = self.unpaired_green_offset * np.array([np.cos(right_angle), np.sin(right_angle)])
                midpoint = green_loc + offset_vec
            
            all_preliminary_midpoints.append(midpoint)

        # Second pass: compute actual midpoints and headings using all preliminary midpoints
        for i, (red_loc, green_loc, is_virtual, red_key, green_key) in enumerate(gate_info):
            # Index into all_preliminary_midpoints includes passed gates
            full_idx = len(self.passed_gate_midpoints) + i
            
            next_midpoint = all_preliminary_midpoints[full_idx + 1] if full_idx + 1 < len(all_preliminary_midpoints) else None
            prev_midpoint = all_preliminary_midpoints[full_idx - 1] if full_idx > 0 else None
            current_prelim_midpoint = all_preliminary_midpoints[full_idx]
            
            midpoint, heading = compute_gate_midpoint_and_heading(
                red_loc=red_loc,
                green_loc=green_loc,
                next_midpoint=next_midpoint,
                prev_midpoint=prev_midpoint,
                current_midpoint=current_prelim_midpoint,
                is_virtual=is_virtual,
            )
            
            self.gate_sequence.append((midpoint, heading, is_virtual))
            
            # Unique gate identifier includes buoy keys to track which buoys were used
            gid = (
                tuple(np.round(midpoint, 3)),
                red_key if red_key else None,
                green_key if green_key else None
            )
            self.gate_ids.append(gid)

        # Reset current_gate_idx to first gate since we filtered out passed ones
        self.current_gate_idx = 0

        self.get_logger().info(
            f"Computed {len(self.gate_sequence)} gates "
            f"({len([g for g in gate_info if not g[2]])} paired, "
            f"{len([g for g in gate_info if g[2]])} virtual). "
            f"Used buoys: {len(self.used_red_buoys)} red, {len(self.used_green_buoys)} green. "
            f"Passed gates: {len(self.passed_gate_midpoints)}"
        )

    def pose_callback(self, msg):
        """Callback to update the boat's current pose.

        Args:
            msg (Float32MultiArray): ROS message containing [x, y, heading] of the boat.
        """
        self.current_pose['x'] = msg.data[0]
        self.current_pose['y'] = msg.data[1]
        self.current_pose['heading'] = msg.data[2]

        if not self.pose_received:
            self.pose_received = True
            self.get_logger().info(
                f"Received first boat pose: ({self.current_pose['x']:.2f}, {self.current_pose['y']:.2f})"
            )

    # ------------------ Main update ------------------
    def try_update_goal(self):
        """Compute and publish the local goal for the boat.

        This function:
        - Checks if the first boat pose has been received
        - Publishes gate midpoints and headings
        - Updates the current gate index
        - Publishes the local goal PoseStamped message
        """
        if not self.pose_received:
            self.get_logger().warn("Waiting for first /boat_pose message...", throttle_duration_sec=2.0)
            return
        
        if not self.gate_sequence:
            if not self.buoys_received:
                self.get_logger().warn("No buoy data received yet...", throttle_duration_sec=2.0)
            return

        # Publish midpoints + headings continuously
        if self.gate_sequence:
            msg = Float32MultiArray()
            msg.data = []
            for midpoint, heading, is_virtual in self.gate_sequence:
                msg.data.extend([midpoint[0], midpoint[1], heading])
            self.midpoints_pub.publish(msg)

        # Determine current gate
        if self.current_gate_idx < len(self.gate_sequence):
            midpoint, heading, is_virtual = self.gate_sequence[self.current_gate_idx]
        else:
            midpoint, heading, is_virtual = self.gate_sequence[-1]

        # Check if gate passed
        gate_id = self.gate_ids[self.current_gate_idx]
        if gate_passed(midpoint, self.current_pose, tol=self.waypoint_pass_tol) and gate_id not in self.passed_gates:
            self.passed_gates.add(gate_id)
            
            # Store the midpoint of the passed gate for future heading calculations
            self.passed_gate_midpoints.append(midpoint.copy())
            
            # Mark buoys as used
            _, red_key, green_key = gate_id
            if red_key:
                self.used_red_buoys.add(red_key)
            if green_key:
                self.used_green_buoys.add(green_key)
            
            gate_type = "virtual" if is_virtual else "paired"
            self.get_logger().info(
                f"Gate {self.current_gate_idx} ({gate_type}) passed. "
                f"Marked buoys as used: red={red_key}, green={green_key}. "
                f"Stored midpoint: {midpoint}"
            )
            
            if self.current_gate_idx < len(self.gate_sequence) - 1:
                self.current_gate_idx += 1
                # Trigger recomputation to exclude used buoys
                self._update_gates()
            else:
                self.get_logger().info("All gates passed!")

        # Compute goal
        if self.current_gate_idx < len(self.gate_sequence):
            goal_pos, gate_heading_stored, _ = self.gate_sequence[self.current_gate_idx]

            # Heading toward the gate if it's the next gate
            heading_vec = np.array(goal_pos) - np.array([self.current_pose['x'], self.current_pose['y']])
            if np.linalg.norm(heading_vec) > 0.001:
                heading = atan2(heading_vec[1], heading_vec[0])
            else:
                heading = gate_heading_stored  # fallback to stored heading
        else:
            # All gates passed → lookahead along last heading
            lookahead = 1.0
            last_heading = self.gate_sequence[-1][1]
            goal_pos = np.array([self.current_pose['x'], self.current_pose['y']]) + \
                    lookahead * np.array([cos(last_heading), sin(last_heading)])
            heading = last_heading

        # Publish goal
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'
        goal_msg.pose.position.x = float(goal_pos[0])
        goal_msg.pose.position.y = float(goal_pos[1])
        goal_msg.pose.orientation.z = sin(heading / 2.0)
        goal_msg.pose.orientation.w = cos(heading / 2.0)
        self.goal_pub.publish(goal_msg)

        self.get_logger().debug(
            f"Goal: gate {self.current_gate_idx} at ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}), "
            f"heading {np.degrees(heading):.1f}°"
        )


# ------------------- Helpers -------------------
def gate_passed(gate_midpoint, current_pose, tol=0.07):
    """Determine if the boat has passed a gate.

    Args:
        gate_midpoint (array-like): [x, y] position of the gate midpoint.
        current_pose (dict): Boat's current pose {'x': float, 'y': float, 'heading': float}.
        tol (float, optional): Distance tolerance to consider gate as passed.

    Returns:
        bool: True if gate is passed, False otherwise.
    """
    gate_vec = np.array(gate_midpoint) - np.array([current_pose['x'], current_pose['y']])
    heading_vec = np.array([cos(current_pose['heading']), sin(current_pose['heading'])])
    distance = np.linalg.norm(gate_vec)
    dot_val = np.dot(gate_vec, heading_vec)

    dot_check = dot_val < 0
    dist_check = distance < 2 * tol

    # Check if passed
    if dist_check:
        return True
    elif dot_check or distance < 2 * tol:
        return True
    else:
        return False


# Main
def main(args=None):
    """Entry point for the local_goal_selector node."""
    rclpy.init(args=args)
    node = LocalGoalSelector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()