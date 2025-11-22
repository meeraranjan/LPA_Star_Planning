# map_publisher.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import OccupancyGrid, MapMetaData
import numpy as np

class LocalMapPublisher(Node):
    def __init__(self):
        super().__init__('local_map_publisher')

        # Subscribers
        self.create_subscription(Float32MultiArray, '/red_buoys', self.red_callback, 10)
        self.create_subscription(Float32MultiArray, '/green_buoys', self.green_callback, 10)
        self.create_subscription(Float32MultiArray, '/dynamic_obstacle', self.dynamic_obstacle_callback, 10)

        # Publisher
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        # Map config
        self.map_width = 100
        self.map_height = 100
        self.map_resolution = 0.1  # meters per cell
        self.origin = [0.0, 0.0]

        # Internal state
        self.red_buoys = []
        self.green_buoys = []
        self.dynamic_obstacles = []

        # Timer
        self.create_timer(0.1, self.publish_map)

    # ------------------ Callbacks ------------------
    def red_callback(self, msg):
        """Parse incoming red buoys: [x0, y0, x1, y1, ...]"""
        arr = msg.data
        n = len(arr) // 2
        self.red_buoys = [(arr[2*i], arr[2*i+1]) for i in range(n)]

    def green_callback(self, msg):
        """Parse incoming green buoys: [x0, y0, x1, y1, ...]"""
        arr = msg.data
        n = len(arr) // 2
        self.green_buoys = [(arr[2*i], arr[2*i+1]) for i in range(n)]

    def dynamic_obstacle_callback(self, msg):
        if len(msg.data) >= 2:
            x, y = msg.data[:2]
            self.dynamic_obstacles.append((x, y))
            self.get_logger().info(f"Added dynamic obstacle at ({x:.2f}, {y:.2f})")

    # ------------------ Publish map ------------------
    def publish_map(self):
        grid = np.zeros((self.map_height, self.map_width), dtype=np.int8)

        # Mark buoys
        for x, y in self.red_buoys + self.green_buoys + self.dynamic_obstacles:
            j = int((x - self.origin[0]) / self.map_resolution)
            i = int((y - self.origin[1]) / self.map_resolution)

            if 0 <= i < self.map_height and 0 <= j < self.map_width:
                grid[i, j] = 100
            else:
                self.get_logger().warn(f"Point ({x:.2f}, {y:.2f}) -> grid ({i}, {j}) out of bounds!")

        # Build OccupancyGrid message
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info = MapMetaData()
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.resolution = self.map_resolution
        msg.info.origin.position.x = self.origin[0]
        msg.info.origin.position.y = self.origin[1]
        msg.info.origin.orientation.w = 1.0

        msg.data = grid.flatten().tolist()
        self.map_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LocalMapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
