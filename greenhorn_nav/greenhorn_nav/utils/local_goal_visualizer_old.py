import matplotlib
print(matplotlib.get_backend())

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from geometry_msgs.msg import PoseStamped

class LocalGoalVisualizer(Node):
    def __init__(self):
        super().__init__('local_goal_visualizer')

        # Data
        self.red_buoys = []
        self.green_buoys = []
        self.midpoints = []
        self.boat_pose = None
        self.goal_pose = None

        # Markers
        self.red_buoy_markers = []
        self.green_buoy_markers = []
        self.midpoint_markers = []
        self.arrow_markers = []

        # Subscriptions
        self.create_subscription(Float32MultiArray, '/buoy_locations', self.buoy_callback, 10)
        self.create_subscription(Float32MultiArray, '/gate_midpoints', self.midpoints_callback, 10)
        self.create_subscription(Float32MultiArray, '/boat_pose', self.pose_callback, 10)
        self.create_subscription(PoseStamped, '/local_goal', self.goal_callback, 10)

        # Setup figure
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-1, 10)
        self.ax.set_ylim(-1, 10)
        self.ax.set_aspect('equal')

        # Boat and goal markers
        self.boat_marker = patches.Circle((0, 0), 0.1, color='blue')
        self.goal_marker = patches.Circle((0, 0), 0.1, color='red')
        self.ax.add_patch(self.boat_marker)
        self.ax.add_patch(self.goal_marker)

    # ---------------- Callbacks ----------------
    def buoy_callback(self, msg):
        if msg.data:
            data = np.array(msg.data)
            n_each = len(data) // 4  # Half red, half green
            self.red_buoys = [(data[i*2], data[i*2+1]) for i in range(n_each)]
            self.green_buoys = [(data[n_each*2 + i*2], data[n_each*2 + i*2 + 1]) for i in range(n_each)]
        else:
            self.red_buoys = []
            self.green_buoys = []

        self.update_plot()

    def midpoints_callback(self, msg):
        if msg.data:
            data = np.array(msg.data).reshape(-1, 3)  # [x, y, heading]
            self.midpoints = data
        else:
            self.midpoints = []
        self.update_plot()

    def pose_callback(self, msg):
        self.boat_pose = msg.data
        self.update_plot()

    def goal_callback(self, msg: PoseStamped):
        self.goal_pose = [msg.pose.position.x, msg.pose.position.y]
        self.update_plot()

    # ---------------- Plotting ----------------
    def update_plot(self):
        # Update boat and goal
        if self.boat_pose:
            self.boat_marker.center = (self.boat_pose[0], self.boat_pose[1])
        if self.goal_pose:
            self.goal_marker.center = (self.goal_pose[0], self.goal_pose[1])

        # Remove old markers
        for patch in self.red_buoy_markers + self.green_buoy_markers + self.midpoint_markers + self.arrow_markers:
            patch.remove()
        self.red_buoy_markers = []
        self.green_buoy_markers = []
        self.midpoint_markers = []
        self.arrow_markers = []

        # Draw buoys
        for rb in self.red_buoys:
            c = patches.Circle(rb, 0.08, color='red')
            self.ax.add_patch(c)
            self.red_buoy_markers.append(c)
        for gb in self.green_buoys:
            c = patches.Circle(gb, 0.08, color='green')
            self.ax.add_patch(c)
            self.green_buoy_markers.append(c)

        # Draw midpoints with arrows
        for mp in self.midpoints:
            x, y, heading = mp
            c = patches.Circle((x, y), 0.05, color='orange')
            self.ax.add_patch(c)
            self.midpoint_markers.append(c)

            arrow_len = 0.2
            dx = arrow_len * np.cos(heading)
            dy = arrow_len * np.sin(heading)
            arr = self.ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.08, fc='orange', ec='orange')
            self.arrow_markers.append(arr)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


# ----- Main -------
def main(args=None):
    rclpy.init(args=args)
    node = LocalGoalVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
