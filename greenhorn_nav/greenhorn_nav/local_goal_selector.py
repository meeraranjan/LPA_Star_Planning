# greenhorn_nav/local_goal_selector.py

import rclpy
from rclpy.node import Node
import numpy as np
from math import cos, sin, atan2
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from greenhorn_nav.utils.buoys2gates import buoys2gates  # Make sure utils is in package

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
        # Timer
        self.timer = self.create_timer(0.5, self.try_update_goal)

        # State
        self.current_pose = {'x': 0.0, 'y': 0.0, 'heading': 0.0}
        self.pose_received = False
        self.gate_sequence = []       # list of (midpoint, heading)
        self.gate_ids = []            # unique identifiers for gates
        self.current_gate_idx = 0

        # Track passed gates persistently
        self.passed_gates = set()  

        # Subscriptions
        self.red_dict = {}
        self.green_dict = {}
        self.buoys_received = False

        self.create_subscription(Float32MultiArray, '/buoy_locations', self.buoy_callback, 10)
        self.create_subscription(Float32MultiArray, '/boat_pose', self.pose_callback, 10)

    # ------------------ Callbacks ------------------
    def buoy_callback(self, msg):
        """Parse incoming buoy locations of the form:
        [r0x, r0y, g0x, g0y, r1x, r1y, g1x, g1y, ...]
        """
        data = msg.data
        n = len(data) // 4  # each gate has 4 numbers

        self.red_dict = {}
        self.green_dict = {}

        for i in range(n):
            r_x = data[4*i + 0]
            r_y = data[4*i + 1]
            g_x = data[4*i + 2]
            g_y = data[4*i + 3]

            self.red_dict[f"r{i}"]   = {"loc": [r_x, r_y]}
            self.green_dict[f"g{i}"] = {"loc": [g_x, g_y]}

        self.buoys_received = True
        self.gate_sequence = []
        self.gate_ids = []

        # Recompute gates
        G, gate_matches, scores = buoys2gates(self.red_dict, self.green_dict)

        # First pass: compute all midpoints
        midpoints = []
        for match in gate_matches:
            red_loc = np.array(G.nodes[match[0]]['loc'])
            green_loc = np.array(G.nodes[match[1]]['loc'])
            midpoint = (red_loc + green_loc) / 2.0
            midpoints.append((midpoint, red_loc, green_loc))
        
        # Second pass: compute headings pointing to next gate
        for i, (midpoint, red_loc, green_loc) in enumerate(midpoints):
            if i + 1 < len(midpoints):
                # Heading points to next gate's midpoint
                next_midpoint = midpoints[i + 1][0]
                vec_to_next = next_midpoint - midpoint
                heading = atan2(vec_to_next[1], vec_to_next[0])
                self.get_logger().debug(
                    f"Gate {i}: heading toward next gate = {np.degrees(heading):.1f}°"
                )
            else:
                # Last gate: use perpendicular bisector
                heading = gate_heading(red_loc, green_loc)
                self.get_logger().debug(
                    f"Gate {i} (last): using perpendicular bisector = {np.degrees(heading):.1f}°"
                )
            
            self.gate_sequence.append((midpoint, heading))

            # Unique gate identifier (based on positions)
            gid = tuple(np.round(midpoint, 3))
            self.gate_ids.append(gid)

        # Update current_gate_idx to first unpassed gate
        for idx, gid in enumerate(self.gate_ids):
            if gid not in self.passed_gates:
                self.current_gate_idx = idx
                break
        else:
            self.current_gate_idx = len(self.gate_ids) - 1

        self.get_logger().info(
            f"Received {n} gates. Current gate index set to {self.current_gate_idx}"
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
        - Computes gate midpoints and headings if not done
        - Updates the current gate index
        - Publishes the local goal PoseStamped message
        """
        if not self.pose_received:
            self.get_logger().warn("Waiting for first /boat_pose message...")
            return
        if not self.gate_sequence:
            if not self.buoys_received:
                self.get_logger().warn("No buoy data received yet...")
                return

        # Publish midpoints + headings continuously
        if self.gate_sequence:
            msg = Float32MultiArray()
            msg.data = []
            for midpoint, heading in self.gate_sequence:
                msg.data.extend([midpoint[0], midpoint[1], heading])
            self.midpoints_pub.publish(msg)

        if not self.gate_sequence:
            self.get_logger().warn("No gates found.")
            return

        # Determine current gate
        if self.current_gate_idx < len(self.gate_sequence):
            midpoint, heading = self.gate_sequence[self.current_gate_idx]
        else:
            midpoint, heading = self.gate_sequence[-1]

        # Check if gate passed
        gate_id = self.gate_ids[self.current_gate_idx]
        if gate_passed(midpoint, self.current_pose) and gate_id not in self.passed_gates:
            self.passed_gates.add(gate_id)
            self.get_logger().info(
                f"Gate {self.current_gate_idx} passed at pose "
                f"x={self.current_pose['x']:.3f}, y={self.current_pose['y']:.3f}"
            )
            if self.current_gate_idx < len(self.gate_sequence) - 1:
                # advance to next unpassed gate
                for idx in range(self.current_gate_idx+1, len(self.gate_ids)):
                    if self.gate_ids[idx] not in self.passed_gates:
                        self.current_gate_idx = idx
                        break
                else:
                    self.current_gate_idx = len(self.gate_sequence) - 1

        # Compute goal
        # Determine heading
        if self.current_gate_idx < len(self.gate_sequence):
            goal_pos, gate_heading_stored = self.gate_sequence[self.current_gate_idx]

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

        self.get_logger().info(
            f"Goal: gate {self.current_gate_idx} at ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}), "
            f"heading {np.degrees(heading):.1f}°"
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
    dist_check = distance < 2* tol

    # Debug messages
    if dist_check:
        print(f"[GateCheck] Passed: within tolerance (distance={distance:.3f}, tol={tol})")
        return True
    elif dot_check or distance < 2 * tol:  # slightly relaxed closeness condition if overshooting
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
