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
# import matplotlib.pyplot as plt
# from PIL import Image

from pid_controller import PositionPIDController, AnglePIDController
from server_wrapper import send_request

class NavigationState(Enum):
    EXPLORATION = 1
    NAVIGATION = 2
    STOP = 3

class PIDState(Enum):
    START = 1
    CONTINUE = 2
    END = 3

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

        # Grounding SAM for object detection and segmentation
        self.grounding_sam = GDSAMClient()

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
        self.angle_pid = AnglePIDController(kp=2.0, ki=0.05, kd=0.1, dt=0.1, max_angular_velocity=1.0)
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

        # Linear PID Controller variables
        self.linear_step = 1
        self.object_distance_threshold = 2.5
        self.linear_step_threshold = 0.05
        self.is_beeline_enabled = False

        # Parameters related to initial exploration rotation. 
        self.max_exploration_rotate_step_count = int(2 * np.pi/ self.angle_step) + 1
        self.exploration_rotate_step_count = 0

        # Create object detection class type
        self.target_object = "forklift."

        # Beeline-parameters
        self.bbox = None
        self.is_depth_region_saved = False
        
        # Create output directory for saved images
        self.output_dir = f"detections/{self.target_object}"
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
        # print(f"Current Pose: {self.current_pose}")

        print("Exploration Rotate Step Count: ", self.exploration_rotate_step_count)

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
                self.step_end_coordinates = None

                # Reset PID terms
                self.angle_pid.prev_error = 0.0
                self.angle_pid.integral = 0.0

                # img = Image.fromarray(self.current_image)
                # img.save(f"{self.output_dir}/input_{ts}.png")
                # response = response.json()

            
                # print(response)
                # print(type(response))
                # with open(f"{self.output_dir}/response_{ts}.json", "w") as f:
                #     json.dump(response, f, indent=4)


                #     # print("Current Image type: ", type(self.current_image))conda 
                #     detections = self.grounding_sam.grounded_segmentation(
                #         Image.fromarray(self.current_image), 
                #         [self.target_object]
                #     )

                #     annotated_image = gd_sam_annotate(self.current_image, detections)
                #     plt.imsave(f"{self.output_dir}/{ts}.png", annotated_image)


                if not self.is_sim_started:

                    self.is_sim_started = True
                    self.sim_start_orientation = self.current_pose.pose.orientation

                    angular_velocity = self.angle_pid.compute_angular_velocity(
                        goal_angle=self.angle_step,
                        curr_angle=0.0
                    )

                    self.pid_state = PIDState.CONTINUE

                elif self.exploration_rotate_step_count == self.max_exploration_rotate_step_count:
                    # Check whether rotated entire 360 
                    # if rotated move to navigation stage. 
                    # FIXME: This part needed to be improved.

                    self.navigation_state = NavigationState.NAVIGATION
                    self.pid_state = PIDState.START
                    angular_velocity = 0.0
                
                else:
                    angular_velocity = self.angle_pid.compute_angular_velocity(
                        goal_angle=self.angle_step,
                        curr_angle=0.0
                    )

                    self.pid_state = PIDState.CONTINUE

                

                if self.current_image is not None:
                    annotated_image = self.current_image.copy()
                    # Save the current image for debugging.
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    response = self.grounding_sam.detections(image=np.array(self.current_image), target_prompt=self.target_object)
                    scores = response["response"]["scores"]
                    bboxes = response["response"]["boxes"]

                    if len(scores) > 0 and max(scores) > 0.8:
                        max_index = scores.index(max(scores))
                        self.bbox = bboxes[max_index]
                        self.navigation_state = NavigationState.NAVIGATION
                        self.pid_state = PIDState.START
                        self.is_beeline_enabled = True
                        angular_velocity = 0.0

                        # Save the depth region for debugging
                        self.is_depth_region_saved = True

                        # Draw bbox on the image
                        x1, y1, x2, y2 = map(int, self.bbox)
                        
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(annotated_image, f"{self.target_object} {scores[max_index]:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

                    cv2.imwrite(f"{self.output_dir}/input_{ts}.png", annotated_image)


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

                print(f"Step angle: {self.angle_step}")
                print(f"Current angle: {current_angle}")

                angular_velocity = self.angle_pid.compute_angular_velocity(
                    goal_angle=self.angle_step,
                    curr_angle=current_angle
                )
                
                # Check this logic? Why not changing
                angle_diff = abs(self.angle_step - current_angle)
                print("Angle Difference: ",  angle_diff)
                if angle_diff < self.rotate_step_threshold:
                    self.pid_state = PIDState.END
                    angular_velocity = 0.0

                    self.step_end_coordinates = np.array([self.current_pose.pose.position.x, self.current_pose.pose.position.y])
                

                cmd.angular.z = angular_velocity

            elif self.pid_state == PIDState.END:
                # Stop the robot 
                # Change the state to START
                cmd = self.get_null_twist()
                cmd.linear.x = 1.0 
                current_coordinates = np.array([self.current_pose.pose.position.x, self.current_pose.pose.position.y])

                distance = np.linalg.norm(current_coordinates - self.step_end_coordinates)
                if distance > 0.25:
                    self.pid_state = PIDState.START

        elif self.navigation_state == NavigationState.NAVIGATION:

            cmd = self.get_null_twist()

            if self.pid_state == PIDState.START:

                programme_flow = []

                print("Block 0")

                if self.is_beeline_enabled:

                    obstacle_region = self.current_depth_image[self.bbox[1]:self.bbox[3], self.bbox[0]:self.bbox[2]]
                    horizontal_aperture_mm = 20.955
                    focal_length_mm = 15.0
                    fov = 2 * math.atan(horizontal_aperture_mm / (2 * focal_length_mm))
                    print("FOV: ", np.degrees(fov))

                    # image shape is by default (480, 640, 3)
                    object_center = [(self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2]
                    print("Image Shape: ", self.current_image.shape)
                    print("Object center: ", object_center)

                    distance = np.median(obstacle_region)
                    print("Object distance (m): ", distance)

                    self.step_start_position = np.array([self.current_pose.pose.position.x, self.current_pose.pose.position.y])
                    print("Step start position: ", self.step_start_position)

                    self.robot_initial_yaw = self.current_pose.pose.orientation.z
                    print("Current orientation: ", self.robot_initial_yaw)

                    # FIXME: Target angle??
                    # The -1 is needed to adjust angle to robots x-y frame. x - forward, y - left. 
                    target_angle = (object_center[0] - self.current_image.shape[1] / 2) / self.current_image.shape[1] * fov * -1 
                    print("Target angle (deg): ", np.degrees(target_angle))

                
                    self.target_position = self.step_start_position + np.array([
                        distance * math.cos(target_angle - self.robot_initial_yaw),
                        distance * math.sin(target_angle - self.robot_initial_yaw)
                    ])

                    print("Target position: ", self.target_position)

                    # Just stop to validate
                    # raise Exception("Stop here")

                    self.is_beeline_enabled = False

                    # Robot is alread within the linear step threshold
                    if distance < self.object_distance_threshold:

                        print("Block 1")
                        programme_flow.append("Block 1")

                        self.navigation_state = NavigationState.STOP
                        self.pid_state = PIDState.START

                    else:
                        print("Block 2")
                        programme_flow.append("Block 2")

                        # self.step_target_position = self.step_start_position + np.array([
                        #     self.linear_step * math.cos(target_angle),
                        #     self.linear_step * math.sin(target_angle)
                        # ])


                        # NOTE: For beelinging just at target as the end position
                        self.step_target_position = self.target_position

                        self.linear_velocity = self.position_pid.compute_linear_velocity(
                            target_position=self.step_target_position,
                            current_position=self.step_start_position
                        )

                        self.pid_state = PIDState.CONTINUE

                        cmd.linear.x = self.linear_velocity * math.cos(target_angle)
                        cmd.linear.y = self.linear_velocity * math.sin(target_angle)

                        # Just Stop 

                        print("Linear velocity: ", self.linear_velocity)
                        print("Cmd: ", cmd)
                        
                else:

                    # NOTE: This logic is needed for stepping: Currently not used. 
                    print("Block 3")
                    programme_flow.append("Block 3")
                    
                    self.step_start_position = np.array([self.current_pose.pose.position.x, self.current_pose.pose.position.y]) - self.step_start_abs_position

                    print("Step start position: ", self.step_start_position)
                    
                    distance = np.linalg.norm(self.target_position - self.step_start_position)

                    print("Distance to target: ", distance)

                    if distance < self.object_distance_threshold:
                        print("Block 4")
                        programme_flow.append("Block 4")
                        self.navigation_state = NavigationState.STOP
                        self.pid_state = PIDState.START

                    else:

                        print("Block 5")
                        programme_flow.append("Block 5")

                        target_angle = math.atan2(self.target_position[1] - self.step_start_position[1], self.target_position[0] - self.step_start_position[0])

                        self.step_target_position = self.step_start_position + np.array([
                            self.linear_step * math.cos(target_angle),
                            self.linear_step * math.sin(target_angle)
                        ])


                        self.linear_velocity = self.position_pid.compute_linear_velocity(
                            target_position=self.step_target_position,
                            current_position=self.step_start_position
                        )

                        cmd.linear.x = self.linear_velocity * math.cos(target_angle)
                        cmd.linear.y = self.linear_velocity * math.sin(target_angle)
                        self.pid_state = PIDState.CONTINUE

                with open("/blue/prabhat/duminduaelamurem/wd/isaac_sim/isaac-go2-ros2/transfer/detections/debug.txt", "a") as f:
                    f.write(f"Navstate: {self.navigation_state}, PID State: {self.pid_state}\n")
                    f.write(f"BBOX: {self.bbox}\n")
                    f.write(f"Object center: {object_center}\n")
                    f.write(f"Object distance: {distance}\n")
                    f.write(f"Target angle: {target_angle}\n")
                    f.write(f"Current position: {self.step_start_position}\n")
                    f.write(f"Target position: {self.target_position}\n")
                    f.write(f"Step target position: {self.step_target_position}\n")
                    f.write(f"Programme flow: {programme_flow}\n")
                    # f.write(f"Linear distance: {linear_distance}\n")


            elif self.pid_state == PIDState.CONTINUE:

                programme_flow = []

                print("Block 6")
                programme_flow.append("Block 6")
                cmd = self.get_null_twist()

                

                # Self.step_start_abs_position should be updated according in the start loop
                current_position = np.array([self.current_pose.pose.position.x, self.current_pose.pose.position.y])
                print("Step target position: ", self.step_target_position)
                print("Current position: ", current_position)
                print("Target position: ", self.target_position)

                self.linear_velocity = self.position_pid.compute_linear_velocity(
                    target_position=self.step_target_position,
                    current_position=current_position
                )

                # Need to check this logic again.
                target_angle = math.atan2(self.step_target_position[1] - current_position[1], self.step_target_position[0] - current_position[0])
                print("Target angle: ", target_angle)


                cmd.linear.x = self.linear_velocity * math.cos(target_angle)
                cmd.linear.y = self.linear_velocity * math.sin(target_angle)

                print("Linear velocity: ", self.linear_velocity)
                print("Cmd: ", cmd)
                # print("Step target position: ", self.step_target_position)

                linear_distance = np.linalg.norm(self.step_target_position - current_position)

                print("Linear distance to step target: ", linear_distance)
                print("Object distance threshold: ", self.object_distance_threshold)
                if linear_distance < self.object_distance_threshold:

                    print("Block 7")
                    programme_flow.append("Block 7")
                    self.navigation_state = NavigationState.STOP
                    self.pid_state = PIDState.START

                with open("/blue/prabhat/duminduaelamurem/wd/isaac_sim/isaac-go2-ros2/transfer/detections/debug.txt", "a") as f:
                    f.write(f"Navstate: {self.navigation_state}, PID State: {self.pid_state}\n")
                    f.write(f"Target Position: {self.step_target_position}\n")
                    current_pose = [self.current_pose.pose.position.x, self.current_pose.pose.position.y]
                    f.write(f"Current Position: {current_pose}\n")
                    f.write(f"Linear distance: {linear_distance}\n")
                    f.write(f"Programme flow: {programme_flow}\n")


                


            elif self.pid_state == PIDState.END:
                print("Block 8")

                self.navigation_state = NavigationState.STOP
                self.pid_state = PIDState.START


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

        print("Start yaw: ", start_yaw)
        print("Current yaw: ", current_yaw)

        # Calculate the angle difference
        angle_diff = current_yaw - start_yaw

        # Wrap the angle difference to the range [-pi, pi]
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

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

class GDSAMClient:
    def __init__(self, port:int = 12183):
        self.url = f"http://localhost:{port}/gdsam"

    def detections(self, image: np.ndarray, target_prompt: str):
        print(f"GDSAMClient.detect_and_segment: {image.shape}, {target_prompt}" )
        response = send_request(self.url, image=image, target_prompt=target_prompt)
        return response

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