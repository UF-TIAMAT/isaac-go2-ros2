#!/usr/bin/env python3
# fixed_cmd_vel.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# ===== Simple knobs =====
PUBLISH_RATE_HZ = 30.0

# Pick one: "rotate_in_place", "orbit_radius", "pivot_front", "orbit_icr"
MODE   = "rotate_in_place"

# Common angular speed (rad/s). + = CCW (left), - = CW (right)
OMEGA  = 0.5

# For MODE == "orbit_radius" (nonholonomic-friendly circle)
RADIUS = 1.0  # meters

# For MODE == "pivot_front" (rotate about a point d meters ahead)
DIST_FRONT = 0.5  # meters

# For MODE == "orbit_icr" (instantaneous center of rotation in base_link)
# x forward, y left (REP-103)
X_ICR = 0.0  # meters
Y_ICR = 1.0  # meters
# ========================

class FixedCmdVel(Node):
    def __init__(self):
        super().__init__('fixed_cmd_vel')
        self.pub = self.create_publisher(Twist, '/unitree_go2/cmd_vel', 10)
        self.timer = self.create_timer(1.0 / PUBLISH_RATE_HZ, self.tick)

    def tick(self):
        cmd = Twist()

        if MODE == "rotate_in_place":
            # Spin about robot center
            cmd.angular.z = OMEGA

        elif MODE == "orbit_radius":
            # Circle of radius R to the left: v_x = ω·R, ω_z = ω
            cmd.linear.x  = OMEGA * RADIUS
            cmd.angular.z = OMEGA

        elif MODE == "pivot_front":
            # Pivot about a point d in front: v_y = -ω·d, ω_z = ω
            cmd.linear.y  = -OMEGA * DIST_FRONT
            cmd.angular.z =  OMEGA

        elif MODE == "orbit_icr":
            # General ICR at (X_ICR, Y_ICR) in base_link:
            # v_x = ω·Y_ICR, v_y = -ω·X_ICR, ω_z = ω
            cmd.linear.x  =  OMEGA * Y_ICR
            cmd.linear.y  = -OMEGA * X_ICR
            cmd.angular.z =  OMEGA

        self.pub.publish(cmd)

def main():
    rclpy.init()
    node = FixedCmdVel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Send a couple of zeroed messages to be safe
        stop = Twist()
        for _ in range(5):
            node.pub.publish(stop)
            rclpy.spin_once(node, timeout_sec=0.01)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()