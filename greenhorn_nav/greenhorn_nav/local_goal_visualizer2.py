# local_goal_visualizer_with_path.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from nav_msgs.msg import OccupancyGrid


class LocalGoalVisualizer(Node):
    def __init__(self):
        super().__init__('local_goal_visualizer')

        # Data
        self.red_buoys = []
        self.green_buoys = []
        self.midpoints = []
        self.boat_pose = None
        self.goal_pose = None
        self.path = []

        # Markers
        self.red_buoy_markers = []
        self.green_buoy_markers = []
        self.midpoint_markers = []
        self.arrow_markers = []
        self.path_line = None
        self.map_image = None
        
        self.map_data = None
        self.map_info = None

        # Subscriptions
        self.create_subscription(OccupancyGrid, '/inflated_map', self.map_callback, 10)
        self.create_subscription(Float32MultiArray, '/buoy_locations', self.buoy_callback, 10)
        self.create_subscription(Float32MultiArray, '/gate_midpoints', self.midpoints_callback, 10)
        self.create_subscription(Float32MultiArray, '/boat_pose', self.pose_callback, 10)
        self.create_subscription(PoseStamped, '/local_goal', self.goal_callback, 10)
        self.create_subscription(Path, '/local_path', self.path_callback, 10)

        # Setup figure
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-1, 10)
        self.ax.set_ylim(-1, 10)
        self.ax.set_aspect('equal')
        self.ax.set_title("Local Navigation Visualization")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True, alpha=0.3)

        # Boat and goal markers
        self.boat_marker = patches.Circle((0, 0), 0.1, color='blue', zorder=10)
        self.goal_marker = patches.Circle((0, 0), 0.1, color='red', zorder=10)
        self.ax.add_patch(self.boat_marker)
        self.ax.add_patch(self.goal_marker)

    # ---------------- Callbacks ----------------
    def map_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        occupied_count = np.sum(self.map_data > 50)
        self.get_logger().info(f"Map received: {width}x{height}, occupied cells: {occupied_count}")
        self.update_plot()

    def buoy_callback(self, msg):
        data = np.array(msg.data)
        if len(data) >= 4:
            n_each = len(data) // 4
            self.red_buoys = [(data[i*2], data[i*2+1]) for i in range(n_each)]
            self.green_buoys = [(data[n_each*2 + i*2], data[n_each*2 + i*2 + 1]) for i in range(n_each)]
        else:
            self.red_buoys = []
            self.green_buoys = []
        self.update_plot()

    def midpoints_callback(self, msg):
        if msg.data:
            data = np.array(msg.data).reshape(-1, 3)
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

    def path_callback(self, msg: Path):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.update_plot()

    # ---------------- Plotting ----------------
    def update_plot(self):
        # Draw the inflated occupancy grid
        if self.map_data is not None and self.map_info is not None:
            if self.map_image:
                self.map_image.remove()
                self.map_image = None
            
            extent = [
                self.map_info.origin.position.x,
                self.map_info.origin.position.x + self.map_info.resolution * self.map_info.width,
                self.map_info.origin.position.y,
                self.map_info.origin.position.y + self.map_info.resolution * self.map_info.height
            ]
            
            # Convert occupancy values: -1 (unknown) -> 0.5, 0-100 -> 0-1
            # Then invert so obstacles (100) show as black
            grid_display = np.where(self.map_data == -1, 0.5, self.map_data / 100.0)
            grid_display = 1.0 - grid_display  # Invert: now 0=black (occupied), 1=white (free)
            
            # Display: white=free (0), black=occupied (100), gray=unknown (-1)
            self.map_image = self.ax.imshow(
                grid_display,
                cmap='gray',
                extent=extent,
                origin='lower',
                alpha=0.7,
                vmin=0,
                vmax=1,
                zorder=1
            )
            
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
            c = patches.Circle(rb, 0.08, color='red', zorder=5)
            self.ax.add_patch(c)
            self.red_buoy_markers.append(c)
        for gb in self.green_buoys:
            c = patches.Circle(gb, 0.08, color='green', zorder=5)
            self.ax.add_patch(c)
            self.green_buoy_markers.append(c)

        # Draw midpoints with arrows
        for mp in self.midpoints:
            x, y, heading = mp
            c = patches.Circle((x, y), 0.05, color='orange', zorder=5)
            self.ax.add_patch(c)
            self.midpoint_markers.append(c)

            arrow_len = 0.2
            dx = arrow_len * np.cos(heading)
            dy = arrow_len * np.sin(heading)
            arr = self.ax.arrow(x, y, dx, dy, head_width=0.05, head_length=0.08, 
                               fc='orange', ec='orange', zorder=5)
            self.arrow_markers.append(arr)

        # Draw path
        if self.path_line:
            self.path_line.remove()
            self.path_line = None
        if self.path:
            x_vals, y_vals = zip(*self.path)
            self.path_line, = self.ax.plot(x_vals, y_vals, color='purple', 
                                           linewidth=2, label='LPA* path', zorder=8)

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