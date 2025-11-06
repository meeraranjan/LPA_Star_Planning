#!/usr/bin/env python3
#lpa_planner_node.py
import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, Float32MultiArray
import math
from greenhorn_nav.utils.grid_processing import detect_changed_cells, inflate_obstacles
from greenhorn_nav.utils.lpa_helpers import LPAStar
from greenhorn_nav.utils.path_smoothing import smooth_path


class LPAPlannerNode(Node):
    def __init__(self):
        super().__init__('demo_lpa_planner_dynamic')

        # Subscriptions and Publishers
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_cb, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/local_goal', self.goal_cb, 10)
        self.boat_sub = self.create_subscription(Float32MultiArray, '/boat_pose', self.boat_cb, 10)
        self.path_pub = self.create_publisher(Path, '/local_path', 10)

        # State
        self.map_msg = None
        self.inflated = None
        self.prev_inflated = None
        self.goal = None
        self.boat_pose = None
        self.lpa = None
        self.last_start = None

        # Parameters
        self.declare_parameter('inflation_radius', 0.1)
        self.declare_parameter('cycle_hz', 2.0)
        self.create_timer(1.0 / float(self.get_parameter('cycle_hz').value), self.cycle)

        self.get_logger().info("LPAPlannerNode initialized with debug prints ON.")

    # ---- Map callback ----
    def map_cb(self, msg: OccupancyGrid):
        self.map_msg = msg
        width = msg.info.width
        height = msg.info.height
        res = msg.info.resolution
        arr = np.array(msg.data, dtype=np.int8).reshape((height, width))
        binmap = (arr > 50).astype(np.uint8)
        inflated = inflate_obstacles(binmap,
                                     inflation_radius=self.get_parameter('inflation_radius').value,
                                     resolution=res)
        self.prev_inflated = self.inflated
        self.inflated = inflated
        self.get_logger().info(f"[MapCB] Map received: {width}x{height}, resolution={res}")
        print(f"[MapCB] Inflated obstacles updated, previous map: {self.prev_inflated is not None}")

    # ---- Goal callback ----
    def goal_cb(self, msg: PoseStamped):
        self.goal = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.get_logger().info(f"[GoalCB] New goal received (world): x={self.goal[0]:.3f}, y={self.goal[1]:.3f}")

    # ---- Boat callback ----
    def boat_cb(self, msg: Float32MultiArray):
        if len(msg.data) >= 2:
            self.boat_pose = (float(msg.data[0]), float(msg.data[1]))
            self.get_logger().info(f"[BoatCB] Boat pose received (world): x={self.boat_pose[0]:.3f}, y={self.boat_pose[1]:.3f}")

    # ---- Coordinate transforms ----
    def world_to_map(self, xy):
        if self.map_msg is None or self.inflated is None:
            return None
        ox = self.map_msg.info.origin.position.x
        oy = self.map_msg.info.origin.position.y
        res = self.map_msg.info.resolution
        try:
            j = int((xy[0] - ox) / res)
            i = int((xy[1] - oy) / res)
        except Exception as e:
            print(f"[Transform] Error converting world->map: {e}")
            return None
        if 0 <= i < self.inflated.shape[0] and 0 <= j < self.inflated.shape[1]:
            print(f"[Transform] World {xy} -> Map ({i},{j})")
            return (i, j)
        print(f"[Transform] World {xy} outside map bounds!")
        return None

    def map_to_world(self, ij):
        ox = self.map_msg.info.origin.position.x
        oy = self.map_msg.info.origin.position.y
        res = self.map_msg.info.resolution
        x = ox + ij[1] * res + res / 2.0
        y = oy + ij[0] * res + res / 2.0
        return (x, y)

    # ---- Main loop ----
    def cycle(self):
        if self.inflated is None or self.boat_pose is None or self.goal is None:
            return

        start = self.world_to_map(self.boat_pose)
        goal = self.world_to_map(self.goal)
        if start is None or goal is None:
            return

        # make sure they are tuples of ints
        start = tuple(int(x) for x in start)
        goal = tuple(int(x) for x in goal)

        print(f"[Cycle] Start: {start}, Goal: {goal}")

        # Initialize LPA* if required
        if self.lpa is None:
            try:
                self.lpa = LPAStar(start=start, goal=goal, grid=self.inflated, heuristic='euclidean')
                self.lpa.km = 0.0
                self.lpa.initialize()
                self.last_start = start
                print("[Cycle] LPA* created and initialized.")
            except Exception as e:
                self.get_logger().error(f"[Cycle] Failed to create LPA*: {e}")
                return

        # Detect changed cells and update LPA*
        if self.prev_inflated is not None:
            changed = detect_changed_cells(self.prev_inflated, self.inflated)
            print(f"[Cycle] Changed cells detected: {len(changed)}")
            for c in changed:
                try:
                    self.lpa.update_vertex(c)
                except Exception as e:
                    print(f"[Cycle] update_vertex failed for {c}: {e}")

        # Handle moving start (boat)
        if start != self.lpa.start:
            try:
                delta_h = self.lpa.heuristic(self.last_start, start)
            except Exception:
                delta_h = 0.0
            self.lpa.km = getattr(self.lpa, 'km', 0.0) + delta_h
            self.last_start = start
            self.lpa.start = start
            print(f"[Cycle] Boat moved. LPA* km incremented by {delta_h:.3f}")

        # Handle moving goal: recreate LPA* cleanly
        if self.lpa.goal != goal:
            self.get_logger().info(f"[Cycle] Goal changed {self.lpa.goal} -> {goal}. Recreating LPA*.")
            try:
                self.lpa = LPAStar(start=start, goal=goal, grid=self.inflated, heuristic='euclidean')
                self.lpa.km = 0.0
                self.lpa.initialize()
                self.last_start = start
                print("[Cycle] LPA* recreated and initialized for new goal.")
            except Exception as e:
                self.get_logger().error(f"[Cycle] Failed to recreate LPA*: {e}")
                return

        # Compute shortest path
        try:
            self.lpa.compute_shortest_path()
            print("[Cycle] compute_shortest_path() done.")
        except Exception as e:
            self.get_logger().error(f"[Cycle] compute_shortest_path() failed: {e}")
            return

        # Defensive: check that start and goal have g values
        if self.lpa.g.get(start, np.inf) == np.inf:
            print("[Cycle] start has infinite g after compute -> no path yet")
            return

        # Extract path (start -> goal)
        raw_path = self.extract_path(start, goal)
        if not raw_path:
            self.get_logger().warn("[Cycle] No path found by extraction!")
            return

        # Smooth path defensively
        try:
            n_raw = len(raw_path)
            num_points = max(5, min(20, int(n_raw * 2)))
            smooth = smooth_path(raw_path, smooth_factor=0.6, num_points=num_points)
            # ensure python list
            if hasattr(smooth, 'tolist'):
                try:
                    smooth = smooth.tolist()
                except Exception:
                    pass
        except Exception as e:
            self.get_logger().warn(f"[Cycle] Smoothing failed ({e}), using raw path.")
            smooth = raw_path

        # Debug print
        self.get_logger().info(f"[LPAPlanner] Path has {len(smooth)} waypoints after smoothing.")
        for i, wp in enumerate(smooth):
            self.get_logger().info(f"  {i}: x={wp[0]:.3f}, y={wp[1]:.3f}")

        # Publish
        self.publish_path(smooth)

    # ---- Path extraction (start -> goal) ----
    def extract_path(self, start, goal):
        # If start has no finite g, no path found
        if self.lpa.g.get(start, np.inf) == np.inf:
            print("[ExtractPath] start has infinite g -> unreachable")
            return []

        path = [start]
        cur = start
        max_iter = 20000
        it = 0
        while cur != goal and it < max_iter:
            it += 1
            nbrs = self.lpa.neighbors(cur)
            next_cell = None
            best_val = np.inf
            # choose neighbor that minimizes cost(cur, n) + g[n]
            for n in nbrs:
                g_n = self.lpa.g.get(n, np.inf)
                if g_n == np.inf:
                    continue
                val = self.lpa.cost(cur, n) + g_n
                if val < best_val:
                    best_val = val
                    next_cell = n
            if next_cell is None:
                print("[ExtractPath] No valid next cell (stuck) during extraction.")
                break
            path.append(next_cell)
            cur = next_cell

        if it >= max_iter:
            print("[ExtractPath] Reached max iterations while extracting path.")

        # convert path (map indices) to world coordinates
        world_path = [self.map_to_world(p) for p in path]
        print(f"[ExtractPath] Extracted path length: {len(world_path)}")
        return world_path

    # ---- Publish path ----
    def publish_path(self, waypoints):
        # handle None/empty list or numpy array
        if waypoints is None or len(waypoints) == 0:
            print("[PublishPath] Empty waypoints -> skip publish")
            return
        msg = Path()
        msg.header = Header()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for wp in waypoints:
            ps = PoseStamped()
            ps.pose.position.x = float(wp[0])
            ps.pose.position.y = float(wp[1])
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)
        print(f"[PublishPath] Published {len(waypoints)} waypoints.")

def main(args=None):
    rclpy.init(args=args)
    node = LPAPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
