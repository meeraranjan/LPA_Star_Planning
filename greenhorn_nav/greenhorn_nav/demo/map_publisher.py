# map_publisher.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import OccupancyGrid, MapMetaData
import numpy as np

class LocalMapPublisher(Node):
    def __init__(self):
        super().__init__('local_map_publisher')

        # Subscriptions
        self.create_subscription(Float32MultiArray, '/buoy_locations', self.buoy_callback, 10)
        
        # Map publisher
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        # Map parameters
        self.map_width = 100
        self.map_height = 100
        self.map_resolution = 0.1  # meters per cell
        self.origin = [0.0, 0.0]   # bottom-left corner in world coordinates

        self.red_buoys = []
        self.green_buoys = []
        self.dynamic_obstacles = []  # list of (x, y) tuples

        self.create_subscription(Float32MultiArray, '/dynamic_obstacle', self.dynamic_obstacle_callback, 10)
        # Timer to periodically publish map
        self.create_timer(0.1, self.publish_map)

    #Callbacks
    def buoy_callback(self, msg):
        if msg.data:
            data = np.array(msg.data)
            n_each = len(data) // 4  # Half red, half green
            self.red_buoys = [(data[i*2], data[i*2+1]) for i in range(n_each)]
            self.green_buoys = [(data[n_each*2 + i*2], data[n_each*2 + i*2 + 1]) for i in range(n_each)]
        else:
            self.red_buoys = []
            self.green_buoys = []
    def dynamic_obstacle_callback(self, msg):
        if msg.data and len(msg.data) >= 2:
            x, y = msg.data[:2]
            self.dynamic_obstacles.append((x, y))
            self.get_logger().info(f"Added dynamic obstacle at ({x:.2f}, {y:.2f})")
    # Map Publishing 
    def publish_map(self):
        grid = np.zeros((self.map_height, self.map_width), dtype=np.int8)
        num_occupied = 0
        # Mark buoys as obstacles
        for x, y in self.red_buoys + self.green_buoys:
            # Convert world coordinates to grid indices
            # j is column (x-direction), i is row (y-direction)
            j = int((x - self.origin[0]) / self.map_resolution)
            i = int((y - self.origin[1]) / self.map_resolution)
            
            if 0 <= i < self.map_height and 0 <= j < self.map_width:
                grid[i, j] = 100  # occupied
                num_occupied += 1
            else:
                self.get_logger().warn(f"Buoy at ({x}, {y}) -> grid ({i}, {j}) is out of bounds!")
        for x, y in self.dynamic_obstacles:
            j = int((x - self.origin[0]) / self.map_resolution)
            i = int((y - self.origin[1]) / self.map_resolution)
            
            if 0 <= i < self.map_height and 0 <= j < self.map_width:
                grid[i, j] = 100  # occupied
            else:
                self.get_logger().warn(f"Dynamic obstacle at ({x}, {y}) -> grid ({i}, {j}) is out of bounds!")
        # if num_occupied > 0:
        #     self.get_logger().info(f"Publishing map with {num_occupied} occupied cells")


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
