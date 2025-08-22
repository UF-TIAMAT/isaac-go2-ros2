import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import requests
import base64
import numpy as np
import time
import json
import os
from datetime import datetime
from geometry_msgs.msg import PoseStamped
import math

class Controller(Node):
    def __init__(self):
        super().__init__('go2_blip2_controller')
        
        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/unitree_go2/cmd_vel', 10)
        self.image_sub = self.create_subscription(
            Image, 
            '/unitree_go2/front_cam/color_image', 
            self.image_callback, 
            10
        )
        self.depth_image_sub = self.create_subscription(
            Image, 
            '/unitree_go2/front_cam/depth_image',
            self.depth_image_callback, 
            10
        )
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/unitree_go2/pose',
            self.pose_callback,
            10
        )

        self.current_pose = None
        self.initial_position = None
        self.target_distance = None
        self.is_moving_forward = False
        self.forward_velocity = 0.8  

        # CV Bridge for image conversion
        self.bridge = CvBridge()
        

        # Control variables
        self.current_image = None
        self.current_depth_image = None
        self.forklift_detected = False
        self.is_moving = False
        self.forklift_bbox = None
        
        # Movement parameters
        self.angular_velocity = 0.5  # rad/s for rotation
        self.detection_interval = 2.0  # seconds between detections
        
        # Create output directory for saved images
        self.output_dir = "forklift_detections"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # # Timer for periodic forklift detection
        # self.detection_timer = self.create_timer(
        #     self.detection_interval, 
        #     self.check_for_forklift
        # )
        
        # # Timer for movement control
        # self.movement_timer = self.create_timer(0.1, self.movement_control)
        
        self.get_logger().info("Go-2 Forklift Controller initialized")
        self.get_logger().info("Starting rotation to search for forklift...")

    def image_callback(self, msg):
        """Callback for receiving camera images"""
        try:
            # Convert ROS image to OpenCV format
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def depth_image_callback(self, msg):
        """Callback for receiving depth images"""
        try:
            # Convert ROS depth image to OpenCV format (typically 16-bit or 32-bit)
            self.current_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
        except Exception as e:
            self.get_logger().error(f"Error converting depth image: {e}")

    def pose_callback(self, msg):
        """Callback for receiving robot pose"""
        self.current_pose = msg

    def stop_robot(self):
        """Stop the robot completely"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.0
        
        # Send stop command multiple times to ensure it's received
        for _ in range(10):
            self.cmd_vel_pub.publish(cmd)
            time.sleep(0.05)
        
        self.get_logger().info("Robot stopped successfully!")
        self.is_moving = False
        self.is_moving_forward = False  # Reset forward movement flag
        
        # Print final summary
        if self.initial_position is not None and self.current_pose is not None:
            final_distance = self.calculate_travel_distance(
                self.initial_position, 
                self.current_pose.pose.position
            )
            self.get_logger().info(f"📍 Final position reached after traveling {final_distance:.2f} meters")


def main(args=None):
    # Initialize ROS2
    rclpy.init(args=args)
    
    try:
        # Create controller node
        controller = Controller()
        
        # Spin the node
        rclpy.spin(controller)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Cleanup
        if 'controller' in locals():
            controller.stop_robot()
            controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()