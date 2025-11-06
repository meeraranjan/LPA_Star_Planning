# greenhorn_nav/local_goal_selector.py
import rclpy
from rclpy.node import Node
import numpy as np
from math import cos, sin, atan2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Float32MultiArray
from tf_transformations import euler_from_quaternion
from greenhorn_navigation.utils import buoys2gates   

class LocalGoalSelector(Node):
    """
    Subscribes:
      - /localization/odometry (nav_msgs/Odometry)
      - /visualization_marker_array (visualization_msgs/MarkerArray) -- buoys from semantic map

    Publishes:
      - /local_goal (geometry_msgs/PoseStamped) <-- x,y,heading (quaternion)
      - /gate_midpoints (std_msgs/Float32MultiArray) <-- debug: [x,y,yaw, x,y,yaw, ...]
    """

    def __init__(self):
        super().__init__('local_goal_selector')

        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/local_goal', 10)
        self.midpoints_pub = self.create_publisher(Float32MultiArray, '/gate_midpoints', 10)

        # Subscriptions
        self.create_subscription(Odometry, '/localization/odometry', self.odom_callback, 10)
        self.create_subscription(MarkerArray, '/visualization_marker_array', self.marker_callback, 10)

        # Timer to run logic 
        self.create_timer(0.5, self.try_update_goal)

        # Internal state
        self.current_pose = None                 # dict: {x,y,heading}
        self.red_dict = {}                       # {id: {'loc':[x,y]}}
        self.green_dict = {}
        self.gate_sequence = []                  # list of (midpoint (np.array), heading (float))
        self.gates_passed = []                   
        self.current_gate_idx = 0

        self.get_logger().info("LocalGoalSelector started")

    # Callbacks
    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        (_, _, yaw) = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.current_pose = {'x': float(pos.x), 'y': float(pos.y), 'heading': float(yaw)}

    def marker_callback(self, msg: MarkerArray):
        """
        Extract red/green buoys from MarkerArray published by SemanticMapManager.
        SemanticMapManager sets marker.ns = 'buoys' and uses color to denote red/green.
        We'll map markers with roughly red color to red_dict and green color to green_dict.
        """
        red = {}
        green = {}
        for m in msg.markers:
            # only process namespace 'buoys' (robustness)
            if m.ns != 'buoys':
                continue

            mx = m.pose.position.x
            my = m.pose.position.y

            # Colors: check r vs g channels
            r_val = getattr(m.color, 'r', 0.0)
            g_val = getattr(m.color, 'g', 0.0)
            b_val = getattr(m.color, 'b', 0.0)

            if r_val > 0.5 and r_val > g_val:
                # red buoy
                red[f"r{m.id}"] = {'loc': [float(mx), float(my)]}
            elif g_val > 0.5 and g_val > r_val:
                green[f"g{m.id}"] = {'loc': [float(mx), float(my)]}
            else:
                # unknown color; ignore for gate formation
                continue

        # update dictionaries atomically
        self.red_dict = red
        self.green_dict = green

        # clear gate_sequence so it recomputes next timer tick (or you can compute immediately)
        self.gate_sequence = []
        self.gates_passed = []
        self.current_gate_idx = 0

    # --- Core logic ----
    def try_update_goal(self):
        if self.current_pose is None:
            self.get_logger().debug("No odometry yet; waiting for /localization/odometry")
            return

        if self.red_dict and self.green_dict and (not self.gate_sequence):
            try:
                G, matches, scores = buoys2gates(self.red_dict, self.green_dict)
            except Exception as e:
                self.get_logger().error(f"buoys2gates failed: {e}")
                return

            # build gate sequence from matches
            seq = []
            for i, match in enumerate(matches):
                n1, n2 = match
                loc1 = np.array(G.nodes[n1]['loc'])
                loc2 = np.array(G.nodes[n2]['loc'])
                midpoint = (loc1 + loc2) / 2.0

                heading = gate_heading(loc1, loc2)

                # if there's a "next" gate, optionally align heading along centerline to it
                if i + 1 < len(matches):
                    nm1, nm2 = matches[i+1]
                    nloc1 = np.array(G.nodes[nm1]['loc'])
                    nloc2 = np.array(G.nodes[nm2]['loc'])
                    next_mid = (nloc1 + nloc2) / 2.0
                    # direction to next midpoint
                    dir_vec = next_mid - midpoint
                    if np.linalg.norm(dir_vec) > 1e-6:
                        heading = atan2(dir_vec[1], dir_vec[0])

                seq.append((midpoint, float(heading)))

            self.gate_sequence = seq
            self.gates_passed = [False] * len(self.gate_sequence)
            self.current_gate_idx = 0
            self.get_logger().info(f"Found {len(self.gate_sequence)} gates")

        # Publish midpoints (debug)
        if self.gate_sequence:
            flat = []
            for (mp, hd) in self.gate_sequence:
                flat.extend([float(mp[0]), float(mp[1]), float(hd)])
            mid_msg = Float32MultiArray()
            mid_msg.data = flat
            self.midpoints_pub.publish(mid_msg)

        if not self.gate_sequence:
            # no gates found: fallback -> lookahead straight from current heading
            lookahead = 3.0
            cp = self.current_pose
            goal_pos = np.array([cp['x'], cp['y']]) + lookahead * np.array([cos(cp['heading']), sin(cp['heading'])])
            goal_yaw = cp['heading']
            self.publish_goal(goal_pos, goal_yaw)
            self.get_logger().debug("No gates: publishing forward lookahead goal")
            return

        # Advance current_gate_idx past already passed gates
        while self.current_gate_idx < len(self.gate_sequence) and self.gates_passed[self.current_gate_idx]:
            self.current_gate_idx += 1

        # If we've passed all gates, lookahead along last heading
        if self.current_gate_idx >= len(self.gate_sequence):
            last_heading = self.gate_sequence[-1][1]
            lookahead = 3.0
            cp = self.current_pose
            goal_pos = np.array([cp['x'], cp['y']]) + lookahead * np.array([cos(last_heading), sin(last_heading)])
            goal_yaw = last_heading
            self.publish_goal(goal_pos, goal_yaw)
            self.get_logger().info("All gates passed; publishing forward lookahead on last heading")
            return

        # Otherwise target the current gate midpoint
        target_mid, target_heading = self.gate_sequence[self.current_gate_idx]

        # mark gate passed if boat is beyond it (on the other side)
        if gate_passed(target_mid, self.current_pose):
            self.gates_passed[self.current_gate_idx] = True
            self.get_logger().info(f"Gate {self.current_gate_idx} passed")
            # attempt to move to next gate on next cycle
            return

        # publish goal at gate midpoint with perpendicular heading
        self.publish_goal(target_mid, target_heading)
        self.get_logger().info(f"Goal: gate {self.current_gate_idx} at ({target_mid[0]:.2f},{target_mid[1]:.2f}), heading {np.degrees(target_heading):.1f}°")

   
    def publish_goal(self, pos, yaw):
        # pos: np.array([x,y]) or list; yaw: float radians
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        # quaternion from yaw (assume roll=pitch=0)
        msg.pose.orientation.z = sin(yaw / 2.0)
        msg.pose.orientation.w = cos(yaw / 2.0)
        self.goal_pub.publish(msg)

# ----------------- Helpers -----------------
def gate_heading(loc_a, loc_b):
    """
    Compute heading (yaw) along the perpendicular bisector of the line joining loc_a -> loc_b.
    loc_a, loc_b: numpy arrays [x,y]
    """
    vec = np.array(loc_b) - np.array(loc_a)
    # perpendicular rotated +90 degrees
    perp = np.array([-vec[1], vec[0]])
    if np.linalg.norm(perp) < 1e-6:
        # degenerate: fallback to along vec
        return atan2(vec[1], vec[0])
    return atan2(perp[1], perp[0])

def gate_passed(gate_midpoint, current_pose, tol=0.5):
    """
    Return True if gate_midpoint is behind the boat (i.e., boat has passed it) OR very close.
    current_pose: dict {x,y,heading}
    """
    gp = np.array(gate_midpoint)
    cp = np.array([current_pose['x'], current_pose['y']])
    gate_vec = gp - cp
    heading_vec = np.array([cos(current_pose['heading']), sin(current_pose['heading'])])
    dist = np.linalg.norm(gate_vec)
    # if projection onto forward heading is negative OR distance very small -> passed
    forward_proj = np.dot(gate_vec, heading_vec)
    return (forward_proj < 0.0) or (dist < tol)

# ----------------- Main -----------------
def main(args=None):
    rclpy.init(args=args)
    node = LocalGoalSelector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
