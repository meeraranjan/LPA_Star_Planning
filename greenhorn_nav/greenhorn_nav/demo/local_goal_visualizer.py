# local_goal_visualizer_pygame.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
import pygame
import numpy as np
import threading


class PygameVisualizer(Node):
    def __init__(self):
        super().__init__('local_goal_visualizer')

        # Data
        self.red_buoys = []
        self.green_buoys = []
        self.midpoints = []
        self.boat_pose = None
        self.goal_pose = None
        self.path = []
        self.map_data = None
        self.map_info = None

        # Pygame settings
        self.screen_width = 800
        self.screen_height = 800
        self.world_min = (-1.0, -1.0)
        self.world_max = (12.0, 12.0)
        
        # Thread lock for data access
        self.data_lock = threading.Lock()

        # Subscriptions
        self.create_subscription(OccupancyGrid, '/inflated_map', self.map_callback, 10)
        self.create_subscription(Float32MultiArray, '/buoy_locations', self.buoy_callback, 10)
        self.create_subscription(Float32MultiArray, '/gate_midpoints', self.midpoints_callback, 10)
        self.create_subscription(Float32MultiArray, '/boat_pose', self.pose_callback, 10)
        self.create_subscription(PoseStamped, '/local_goal', self.goal_callback, 10)
        self.create_subscription(Path, '/local_path', self.path_callback, 10)

        # Publisher for dynamic obstacles
        self.dynamic_obstacle_pub = self.create_publisher(Float32MultiArray, '/dynamic_obstacle', 10)

        # Start pygame in separate thread
        self.running = True
        self.pygame_thread = threading.Thread(target=self.run_pygame, daemon=True)
        self.pygame_thread.start()

        self.get_logger().info("Pygame Visualizer initialized")

    # ---------------- Callbacks ----------------
    def map_callback(self, msg: OccupancyGrid):
        with self.data_lock:
            width = msg.info.width
            height = msg.info.height
            self.map_info = msg.info
            self.map_data = np.array(msg.data, dtype=np.int8).reshape((height, width))
            occupied_count = np.sum(self.map_data > 50)

    def buoy_callback(self, msg):
        with self.data_lock:
            data = np.array(msg.data)
            if len(data) >= 4:
                n_each = len(data) // 4
                self.red_buoys = [(data[i*2], data[i*2+1]) for i in range(n_each)]
                self.green_buoys = [(data[n_each*2 + i*2], data[n_each*2 + i*2 + 1]) for i in range(n_each)]
            else:
                self.red_buoys = []
                self.green_buoys = []

    def midpoints_callback(self, msg):
        with self.data_lock:
            if msg.data:
                data = np.array(msg.data).reshape(-1, 3)
                self.midpoints = data.tolist()
            else:
                self.midpoints = []

    def pose_callback(self, msg):
        with self.data_lock:
            self.boat_pose = msg.data

    def goal_callback(self, msg: PoseStamped):
        with self.data_lock:
            self.goal_pose = [msg.pose.position.x, msg.pose.position.y]

    def path_callback(self, msg: Path):
        with self.data_lock:
            self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]

    # ---------------- Coordinate Conversion ----------------
    def world_to_screen(self, x, y):
        wx_min, wy_min = self.world_min
        wx_max, wy_max = self.world_max
        nx = (x - wx_min) / (wx_max - wx_min)
        ny = (y - wy_min) / (wy_max - wy_min)
        sx = int(nx * self.screen_width)
        sy = int((1 - ny) * self.screen_height)
        return sx, sy

    def screen_to_world(self, sx, sy):
        wx_min, wy_min = self.world_min
        wx_max, wy_max = self.world_max
        nx = sx / self.screen_width
        ny = 1 - sy / self.screen_height
        wx = wx_min + nx * (wx_max - wx_min)
        wy = wy_min + ny * (wy_max - wy_min)
        return wx, wy

    def world_to_screen_scale(self, length):
        wx_min, wx_max = self.world_min[0], self.world_max[0]
        return int(length * self.screen_width / (wx_max - wx_min))

    # ---------------- Pygame Rendering ----------------
    def run_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Local Navigation Visualization")
        clock = pygame.time.Clock()

        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GRAY = (128, 128, 128)
        RED = (255, 0, 0)
        GREEN = (0, 255, 0)
        BLUE = (0, 0, 255)
        ORANGE = (255, 165, 0)
        PURPLE = (128, 0, 128)
        DARK_GRAY = (64, 64, 64)

        font = pygame.font.Font(None, 24)

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    world_pos = self.screen_to_world(*mouse_pos)
                    self.add_dynamic_obstacle(world_pos)

            self.screen.fill(WHITE)

            with self.data_lock:
                if self.map_data is not None and self.map_info is not None:
                    self.draw_occupancy_grid(self.screen, DARK_GRAY)

                self.draw_grid(self.screen, GRAY)

                if self.path:
                    self.draw_path(self.screen, PURPLE)

                for rb in self.red_buoys:
                    self.draw_circle(self.screen, RED, rb, 0.08)
                for gb in self.green_buoys:
                    self.draw_circle(self.screen, GREEN, gb, 0.08)

                for mp in self.midpoints:
                    x, y, heading = mp
                    self.draw_circle(self.screen, ORANGE, (x, y), 0.05)
                    self.draw_arrow(self.screen, ORANGE, (x, y), heading, 0.2)

                if self.boat_pose:
                    self.draw_circle(self.screen, BLUE, (self.boat_pose[0], self.boat_pose[1]), 0.1)

                if self.goal_pose:
                    self.draw_circle(self.screen, RED, (self.goal_pose[0], self.goal_pose[1]), 0.1)

                info_lines = [
                    f"Buoys: R={len(self.red_buoys)} G={len(self.green_buoys)}",
                    f"Path waypoints: {len(self.path)}",
                    f"Boat: {self.boat_pose[:2] if self.boat_pose else 'None'}",
                ]
                for i, line in enumerate(info_lines):
                    text = font.render(line, True, BLACK)
                    self.screen.blit(text, (10, 10 + i * 25))

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

    # ---------------- Drawing Helpers ----------------
    def draw_grid(self, screen, color):
        wx_min, wy_min = self.world_min
        wx_max, wy_max = self.world_max
        for x in range(int(wx_min), int(wx_max) + 1):
            sx1, sy1 = self.world_to_screen(x, wy_min)
            sx2, sy2 = self.world_to_screen(x, wy_max)
            pygame.draw.line(screen, color, (sx1, sy1), (sx2, sy2), 1)
        for y in range(int(wy_min), int(wy_max) + 1):
            sx1, sy1 = self.world_to_screen(wx_min, y)
            sx2, sy2 = self.world_to_screen(wx_max, y)
            pygame.draw.line(screen, color, (sx1, sy1), (sx2, sy2), 1)

    def draw_occupancy_grid(self, screen, color):
        if self.map_data is None or self.map_info is None:
            return
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        occupied_cells = np.argwhere(self.map_data > 50)
        for i, j in occupied_cells:
            wx = ox + j * res
            wy = oy + i * res
            sx, sy = self.world_to_screen(wx, wy)
            cell_size = self.world_to_screen_scale(res)
            pygame.draw.rect(screen, color, (sx, sy, cell_size, cell_size))

    def draw_circle(self, screen, color, world_pos, world_radius):
        sx, sy = self.world_to_screen(world_pos[0], world_pos[1])
        radius = self.world_to_screen_scale(world_radius)
        pygame.draw.circle(screen, color, (sx, sy), max(radius, 2))

    def draw_arrow(self, screen, color, world_pos, heading, length):
        x, y = world_pos
        dx = length * np.cos(heading)
        dy = length * np.sin(heading)
        sx1, sy1 = self.world_to_screen(x, y)
        sx2, sy2 = self.world_to_screen(x + dx, y + dy)
        pygame.draw.line(screen, color, (sx1, sy1), (sx2, sy2), 2)
        arrow_angle = 0.5
        arrow_len = self.world_to_screen_scale(length * 0.3)
        angle1 = heading + np.pi - arrow_angle
        angle2 = heading + np.pi + arrow_angle
        ax1 = sx2 + arrow_len * np.cos(angle1)
        ay1 = sy2 - arrow_len * np.sin(angle1)
        ax2 = sx2 + arrow_len * np.cos(angle2)
        ay2 = sy2 - arrow_len * np.sin(angle2)
        pygame.draw.line(screen, color, (sx2, sy2), (int(ax1), int(ay1)), 2)
        pygame.draw.line(screen, color, (sx2, sy2), (int(ax2), int(ay2)), 2)

    def draw_path(self, screen, color):
        if len(self.path) < 2:
            return
        screen_points = [self.world_to_screen(x, y) for x, y in self.path]
        pygame.draw.lines(screen, color, False, screen_points, 3)

    # ---------------- Dynamic Obstacles ----------------
    def add_dynamic_obstacle(self, world_pos):
        msg = Float32MultiArray()
        msg.data = [world_pos[0], world_pos[1]]
        self.dynamic_obstacle_pub.publish(msg)
        self.get_logger().info(f"Dynamic obstacle added at {world_pos}")

    # ---------------- Cleanup ----------------
    def destroy_node(self):
        self.running = False
        if self.pygame_thread.is_alive():
            self.pygame_thread.join(timeout=1.0)
        super().destroy_node()


# ----- Main -------
def main(args=None):
    rclpy.init(args=args)
    node = PygameVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
