#!/usr/bin/env python3
# boat_simulator.py
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from std_msgs.msg import Float32MultiArray
import numpy as np
import math


class BoatSimulator(Node):
    def __init__(self):
        super().__init__('boat_simulator')

        # --- ROS interfaces ---
        self.create_subscription(Path, '/local_path', self.path_cb, 10)
        self.pose_pub = self.create_publisher(Float32MultiArray, '/boat_pose', 10)

        # --- Boat state ---
        self.pos = np.array([0.0, 0.0])   # starting position
        self.heading = 0.0                # radians
        self.path = []
        self.wp_index = 0

        # --- Motion parameters ---
        self.speed = 1.2   # meters per second
        self.dt = 1       # seconds per cycle
        self.waypoint_tolerance = 0.5  # meters, increased from 0.05

        # --- Timer ---
        self.create_timer(self.dt, self.update)
        self.get_logger().info("BoatSimulator started, waiting for /local_path...")

    # ---------------- Path callback ----------------
    def path_cb(self, msg: Path):
        new_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if not new_path:
            self.get_logger().warn("Received empty path.")
            return

        if not self.path:
            # first path received
            self.path = new_path
            self.wp_index = 0
            self.get_logger().info(f"Received initial path with {len(self.path)} waypoints.")
            return

        # find closest point on new path to current position
        dists = [np.linalg.norm(np.array(pt) - self.pos) for pt in new_path]
        closest_index = min(int(np.argmin(dists)), len(new_path)-1)
        self.wp_index = closest_index

        # Only update wp_index if the closest waypoint is far from current index
        if abs(closest_index - self.wp_index) > 3:
            self.wp_index = closest_index

        self.path = new_path
        self.get_logger().info(f"Updated path, continuing from waypoint {self.wp_index}/{len(self.path)-1}")

    # ---------------- Main update loop ----------------
    def update(self):
        if self.wp_index >= len(self.path):
            self.wp_index = len(self.path)-1
            self.publish_pose()
            return
        if not self.path or self.wp_index >= len(self.path):
            self.publish_pose()
            return

        target = np.array(self.path[self.wp_index])
        direction = target - self.pos
        dist = np.linalg.norm(direction)

        # reached this waypoint
        if dist < self.waypoint_tolerance:
            self.wp_index += 1
            if self.wp_index >= len(self.path):
                self.get_logger().info("Reached final waypoint.")
                self.publish_pose()
                return
            target = np.array(self.path[self.wp_index])
            direction = target - self.pos
            dist = np.linalg.norm(direction)

        # move toward target
        if dist > 1e-6:
            direction /= dist
            step = min(self.speed * self.dt, dist)
            self.pos += direction * step
            self.heading = math.atan2(direction[1], direction[0])

        self.publish_pose()

    # ---------------- Publish boat pose ----------------
    def publish_pose(self):
        msg = Float32MultiArray()
        msg.data = [float(self.pos[0]), float(self.pos[1]), float(self.heading)]
        self.pose_pub.publish(msg)
        # self.get_logger().debug(f"[BoatSim] Pose: x={self.pos[0]:.2f}, y={self.pos[1]:.2f}, θ={math.degrees(self.heading):.1f}°")


def main(args=None):
    rclpy.init(args=args)
    node = BoatSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
