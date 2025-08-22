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
from enum import Enum
from scipy.spatial.transform import Rotation as R

from pid_controller import PositionPIDController, AnglePIDController

class NavigationState(Enum):
    EXPLORATION = 1
    NAVIGATION = 2
    STOP = 3

class PIDState(Enum):
    START = 1
    CONTINUE = 2

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
        self.is_sim_started = False
        
        # Movement parameters
        self.angular_velocity = 0.5  # rad/s for rotation
        self.detection_interval = 2.0  # seconds between detections

        # PID Controllers
        self.angle_pid = AnglePIDController(kp=1.0, ki=0.0, kd=0.1, dt=0.1, max_angular_velocity=1.0)
        self.position_pid = PositionPIDController(kp=0.5, ki=0.0, kd=0.05, dt=0.1, max_linear_velocity=1.0)

        # Navigation States
        self.navigation_state = NavigationState.EXPLORATION
        self.pid_state = PIDState.START
        self.target_position = None

        # Angle PID Controller variables
        self.sim_start_orientation = None
        self.angle_step = np.pi/6
        self.step_start_orientation = None
        self.step_current_orientation = None
        self.rotate_step_threshold = 0.01

        # Parameters related to initial exploration rotation. 
        self.max_exploration_rotate_step_count = int(2 * np.pi/ self.angle_step) + 1
        self.exploration_rotate_step_count = 0
        
        # Create output directory for saved images
        self.output_dir = "forklift_detections"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # # Timer for periodic forklift detection
        # self.detection_timer = self.create_timer(
        #     self.detection_interval, 
        #     self.check_for_forklift
        # )
        
        # # Timer for movement control
        self.movement_timer = self.create_timer(0.1, self.movement_control)
        
        self.get_logger().info("Go-2 Forklift Controller initialized")
        self.get_logger().info("Starting rotation to search for forklift...")

    def movement_control(self):
        """Control robot movement"""

        cmd = Twist()

        print(f"Entering State: {self.navigation_state} and PID State: {self.pid_state}")
        print(f"Current Pose: {self.current_pose}")

        # FIXME: Got a current pose none at the beginning. what is the reason?
        if self.current_pose is None:
            return

        if self.navigation_state == NavigationState.EXPLORATION:
            if self.pid_state == PIDState.START:
                # Step 1
                # register the initial rotation/position if the very first time 
                # Set PID rotation target to 30 degree and change state to continue. 
                # Find the angle velocity command. 

                self.step_start_orientation = self.current_pose.pose.orientation
                self.exploration_rotate_step_count += 1

                if not self.is_sim_started:

                    self.is_sim_started = True
                    self.sim_start_orientation = self.current_pose.pose.orientation

                    angular_velocity = self.angle_pid.compute_angular_velocity(
                        goal_angle=self.angle_step,
                        curr_angle=0.0
                    )

                elif self.exploration_rotate_step_count == self.max_exploration_rotate_step_count:
                    # Check whether rotated entire 360 
                    # if rotated move to navigation stage. 

                    self.navigation_state = NavigationState.NAVIGATION
                    self.pid_state = PIDState.START
                    angular_velocity = 0.0
                
                else:
                    angular_velocity = self.angle_pid.compute_angular_velocity(
                        goal_angle=self.angle_step,
                        curr_angle=0.0
                    )

                self.pid_state = PIDState.CONTINUE

                # Set the angular velocity command
                print("Angular Velocity type: ", type(angular_velocity))
                cmd.angular.z = angular_velocity

                # Step 2
                # Add object detection in the start
                # If detected change the state to NAVIGATION

            elif self.pid_state == PIDState.CONTINUE:
                # input current rotation and and goal angle
                # find the angular error 
                # if angular error is small ignore and set to START

                self.step_current_orientation = self.current_pose.pose.orientation

                current_angle = self.get_current_angle(self.step_current_orientation, self.step_start_orientation)

                angular_velocity = self.angle_pid.compute_angular_velocity(
                    goal_angle=self.angle_step,
                    curr_angle=current_angle
                )

                if angular_velocity < self.rotate_step_threshold:
                    self.pid_state = PIDState.START
                    angular_velocity = 0.0


                print("Angular Velocity type: ", type(angular_velocity))
                cmd.angular.z = angular_velocity

        elif self.navigation_state == NavigationState.NAVIGATION:

            if self.pid_state == PIDState.START:
                cmd = self.get_null_twist()

            elif self.pid_state == PIDState.CONTINUE:
                cmd = self.get_null_twist()

        elif self.navigation_state == NavigationState.STOP:
            cmd = self.get_null_twist()

        print(f"Exiting State: {self.navigation_state} and PID State: {self.pid_state}")
        print(f"Twist command: {cmd}")
        self.cmd_vel_pub.publish(cmd)

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

    def get_null_twist(self):
        """Get a null twist message"""

        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        cmd.angular.z = 0.0

        return cmd
    

    def get_yaw_from_orientation(self, orientation):
        """Convert orientation to yaw angle around z-axis"""

        quaternion_orientation = [orientation.x, orientation.y, orientation.z, orientation.w]
        rotation = R.from_quat(quaternion_orientation)

        yaw = rotation.as_euler("zyx", degrees=False)[0]

        return yaw 
    
    def get_current_angle(self, current_orientation, start_orientation):
        """Calculate the current angle based on start orientation and current orientation"""

        start_yaw = self.get_yaw_from_orientation(start_orientation)
        current_yaw = self.get_yaw_from_orientation(current_orientation)

        # Calculate the angle difference
        angle_diff = current_yaw - start_yaw

        return angle_diff

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