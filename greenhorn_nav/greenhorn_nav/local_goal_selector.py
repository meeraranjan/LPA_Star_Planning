# greenhorn_nav/local_goal_selector.py

import rclpy
from rclpy.node import Node
import numpy as np
from math import cos, sin, atan2
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from greenhorn_nav.utils.buoys2gates import buoys2gates  # Make sure utils is in package
from math import cos, sin

class LocalGoalSelector(Node):
    """ROS2 Node that selects local goals for the boat to navigate through gates.

    The node subscribes to the boat's pose, publishes local goals, midpoints of gates,
    and buoy locations. It tracks which gates have been passed and updates the next goal
    accordingly.
    """
    def __init__(self):
        super().__init__('local_goal_selector')

        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/local_goal', 10)
        self.midpoints_pub = self.create_publisher(Float32MultiArray, '/gate_midpoints', 10)
        self.buoy_pub = self.create_publisher(Float32MultiArray, '/buoy_locations', 10)

        # Timer
        self.timer = self.create_timer(0.5, self.try_update_goal)

        # State
        self.current_pose = {'x': 0.0, 'y': 0.0, 'heading': 0.0}
        self.pose_received = False
        self.gate_sequence = []       # list of (midpoint, heading)
        self.gates_passed = []        # 0.0 = not passed, 1.0 = passed
        self.current_gate_idx = 0

        # Subscriptions
        self.create_subscription(Float32MultiArray, '/boat_pose', self.pose_callback, 10)

    # ------------------ Callbacks ------------------
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
            self.get_logger().info(f"Received first boat pose: ({self.current_pose['x']:.2f}, {self.current_pose['y']:.2f})")

    # ------------------ Buoy helpers ------------------
    def get_buoy_dicts(self):
        """Return dictionaries of red and green buoys with their locations.

        Returns:
            tuple: Two dictionaries (red_dict, green_dict) containing buoy locations.
        """
        red_dict = {
            'r0': {'loc': [0.0, 1.0]},
            'r1': {'loc': [2.5, 3.5]},
            'r2': {'loc': [5.0, 6.0]},
            'r3': {'loc': [7.0, 7.5]},
        }
        green_dict = {
            'g0': {'loc': [1.5, 1.0]},
            'g1': {'loc': [4.5, 3.0]},
            'g2': {'loc': [7.5, 5.0]},
            'g3': {'loc': [9.5, 7.5]},
        }
        return red_dict, green_dict

    def publish_buoy_locations(self):
        """Publish the locations of all buoys as a flattened Float32MultiArray message."""
        red_dict, green_dict = self.get_buoy_dicts()
        msg = Float32MultiArray()
        data = []
        for key in sorted(red_dict.keys()):
            data.extend(red_dict[key]['loc'])
        for key in sorted(green_dict.keys()):
            data.extend(green_dict[key]['loc'])
        msg.data = data
        self.buoy_pub.publish(msg)

    # ------------------ Main update ------------------
    def try_update_goal(self):
        """Compute and publish the local goal for the boat.

        This function:
        - Publishes buoy locations
        - Checks if the first boat pose has been received
        - Computes gate midpoints and headings if not done
        - Updates the current gate index
        - Publishes the local goal PoseStamped message
        """
        # Publish buoy locations continuously
        self.publish_buoy_locations()
        if not self.pose_received:
            self.get_logger().warn("Waiting for first /boat_pose message...")
            return
        # Compute gates if not yet done
        if not self.gate_sequence:
            red_dict, green_dict = self.get_buoy_dicts()
            G, gate_matches, scores = buoys2gates(red_dict, green_dict)
            self.gate_sequence = []

            for i, match in enumerate(gate_matches):
                red_loc = np.array(G.nodes[match[0]]['loc'])
                green_loc = np.array(G.nodes[match[1]]['loc'])
                midpoint = (red_loc + green_loc) / 2.0

                # Default heading = perpendicular bisector
                heading = gate_heading(red_loc, green_loc)

                # If there is a next gate, compute heading along centerline
                if i + 1 < len(gate_matches):
                    next_match = gate_matches[i + 1]
                    next_red = np.array(G.nodes[next_match[0]]['loc'])
                    next_green = np.array(G.nodes[next_match[1]]['loc'])
                    next_midpoint = (next_red + next_green) / 2.0
                    vec = next_midpoint - midpoint
                    heading = atan2(vec[1], vec[0])

                self.gate_sequence.append((midpoint, heading))

            self.gates_passed = [0.0] * len(self.gate_sequence)

        # Publish midpoints + headings continuously
        if self.gate_sequence:
            msg = Float32MultiArray()
            msg.data = np.array([[mp[0][0], mp[0][1], mp[1]] for mp in self.gate_sequence]).flatten().tolist()
            self.midpoints_pub.publish(msg)

        if not self.gate_sequence:
            self.get_logger().warn("No gates found.")
            return

        # Determine current gate
        if self.current_gate_idx < len(self.gate_sequence):
            midpoint, heading = self.gate_sequence[self.current_gate_idx]
            # print(f"[LocalGoalSelector] Boat pose: x={self.current_pose['x']:.3f}, y={self.current_pose['y']:.3f}, heading={np.degrees(self.current_pose['heading']):.1f}°")

        else:
            midpoint, heading = self.gate_sequence[-1]

        # Check if gate passed
        if self.current_gate_idx < len(self.gate_sequence) and gate_passed(midpoint, self.current_pose):
            if self.gates_passed[self.current_gate_idx] == 0.0:
                print(f"[LocalGoalSelector] Gate {self.current_gate_idx} passed at pose x={self.current_pose['x']:.3f}, y={self.current_pose['y']:.3f}")
                self.get_logger().info(f"Gate {self.current_gate_idx} passed.")
            self.gates_passed[self.current_gate_idx] = 1.0
            if self.current_gate_idx < len(self.gate_sequence) - 1:
                self.current_gate_idx += 1

        # Compute goal
        if all(self.gates_passed):
            # All gates passed → lookahead along last heading
            lookahead = 1.0
            last_heading = self.gate_sequence[-1][1]
            goal_pos = np.array([self.current_pose['x'], self.current_pose['y']]) + \
                       lookahead * np.array([cos(last_heading), sin(last_heading)])
            heading = last_heading
        else:
            # Skip passed gates
            while self.current_gate_idx < len(self.gate_sequence) and self.gates_passed[self.current_gate_idx]:
                self.current_gate_idx += 1

            if self.current_gate_idx < len(self.gate_sequence):
                goal_pos, heading = self.gate_sequence[self.current_gate_idx]
            else:
                # Edge case: all gates just passed
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

        self.get_logger().info(
            f"Goal: gate {self.current_gate_idx} at ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}), heading {np.degrees(heading):.1f}°"
        )

# ------------------- Helpers -------------------
def gate_heading(red_loc, green_loc):
    """Compute the perpendicular bisector heading of a gate formed by two buoys.

    Args:
        red_loc (array-like): [x, y] position of the red buoy.
        green_loc (array-like): [x, y] position of the green buoy.

    Returns:
        float: Heading angle (radians) perpendicular to the line connecting the buoys.
    """
    vec = np.array(green_loc) - np.array(red_loc)
    perp_vec = np.array([-vec[1], vec[0]])  # perpendicular
    return atan2(perp_vec[1], perp_vec[0])


def gate_passed(gate_midpoint, current_pose, tol=0.07):
    """Determine if the boat has passed a gate.

    Args:
        gate_midpoint (array-like): [x, y] position of the gate midpoint.
        current_pose (dict): Boat's current pose {'x': float, 'y': float, 'heading': float}.
        tol (float, optional): Distance tolerance to consider gate as passed. Defaults to 0.07.

    Returns:
        bool: True if gate is passed, False otherwise.
    """
    
    gate_vec = np.array(gate_midpoint) - np.array([current_pose['x'], current_pose['y']])
    heading_vec = np.array([cos(current_pose['heading']), sin(current_pose['heading'])])
    distance = np.linalg.norm(gate_vec)
    dot_val = np.dot(gate_vec, heading_vec)

    dot_check = dot_val < 0
    dist_check = distance < tol

    # Debug messages
    if dist_check:
        print(f"[GateCheck] Passed: within tolerance (distance={distance:.3f}, tol={tol})")
        return True
    elif dot_check and distance < 2 * tol:  # slightly relaxed closeness condition if overshooting
        print(f"[GateCheck] Passed: moved past gate (dot={dot_val:.3f}) and still near (distance={distance:.3f})")
        return True
    else:
        # print(f"[GateCheck] Not passed yet (dot={dot_val:.3f}, distance={distance:.3f})")
        return False

#  Main 
def main(args=None):
    """Entry point for the local_goal_selector node."""
    rclpy.init(args=args)
    node = LocalGoalSelector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
