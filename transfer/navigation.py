import numpy as np
from transfer.cosine_similarity import HFBLIP2ImageTextRetrieval
from datetime import datetime
from enum import Enum
import torch

import go2.go2_ctrl as go2_ctrl


class NavigationState(Enum):
    EXPLORATION = 1
    NAVIGATION = 2
    STOP = 3

class StateStage(Enum):
    START = 1
    INACTION = 2
    END = 3
    

def vlfm_navigation(rgb: np.ndarray, depth: np.ndarray, prompt: str, navstate: NavigationState, state_stage: StateStage, similarity_model: HFBLIP2ImageTextRetrieval, threshold: float = 0.5, odometry: dict = None) -> tuple(NavigationState, StateStage):

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    velocity_value = 0.25  # Default forward velocity/ angular velocity


    if navstate == NavigationState.EXPLORATION:

        if state_stage == StateStage.START:
            # Initially set up the angle step size. 
            # If cosine similarity is higher, find the object position and move towards it
            # Set state to navigation with (rotation and position)

            rotation_step_count = 0
            max_rotation_steps = 8
            rotation_per_step = 2 * np.pi / max_rotation_steps
            current_angle = 0
            goal_angle = rotation_per_step
            previous_rotation = odometry["rotation"]

            cosine_similarity = similarity_model.cosine_image_text(rgb, prompt)
            # This has to be saved for END logic
            # Find the direction_related_to_robot_frontier()

            if cosine_similarity > threshold:  #object detected
                pass # def_target_rho_theta()
                navstate = NavigationState.NAVIGATION
                state_stage = StateStage.START

            elif cosine_similarity < threshold: # not object detected. 
                state_stage = StateStage.INACTION

        elif state_stage == StateStage.INACTION:

            current_rotation = odometry["rotation"]
            current_angle, previous_angle = 0, 0 #need to find from rotation. get_current_angle(previous_rotation, current_rotation)

            anglular_velocity = 0 # compute using angle PID controller (goal_angle, current_angle)
            angular_velocity_threshold = 0 
            if anglular_velocity < angular_velocity_threshold:
                pass # Should stop continuing rotation. 
                rotation_step_count += 1
                # find cosine similarity
                # find GD-SAM 
                # find the distance_related_to_robot_frontier

                # if object is in GD-SAM send to navigation 
                # else append cosine similarity and continue navigation. 
                #   : Have set a new goal_angle += rotation_per_step 

        elif state_stage == StateStage.END:
            pass

            # Select best direction from the cosine_similarities and distance logic
            # Pass it to navigation. 

        # if cosine_similarity > threshold:
        #     # Move forward in exploration state
        #     go2_ctrl.base_vel_cmd_input[0] = torch.tensor([velocity_value, 0.0, 0.0], dtype=torch.float32)
        #     navstate = NavigationState.NAVIGATION
        # else:
        #     # Continue exploring by turning in place
        #     go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, velocity_value], dtype=torch.float32)

    elif navstate == NavigationState.NAVIGATION:

        left = rgb[:, :int(rgb.shape[1] / 3)]
        middle = rgb[:, int(rgb.shape[1] / 3):int(2 * rgb.shape[1] / 3)]
        right = rgb[:, int(2 * rgb.shape[1] / 3):]

        # Check if the cosine similarity is below the threshold
        if cosine_similarity < threshold:
            left_cosine = similarity_model.cosine_image_text(left, prompt)
            middle_cosine = similarity_model.cosine_image_text(middle, prompt)
            right_cosine = similarity_model.cosine_image_text(right, prompt)

            # Determine the direction to turn based on cosine similarity
            if left_cosine > right_cosine and left_cosine > middle_cosine:
                # Turn left
                go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, velocity_value], dtype=torch.float32)
                # go2_ctrl.base_vel_cmd_input[0] = torch.tensor([velocity_value, velocity_value/10, 0.0], dtype=torch.float32)
            elif right_cosine > left_cosine and right_cosine > middle_cosine:
                # Turn right
                go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, -velocity_value], dtype=torch.float32)
            else:
                # Move forward if middle cosine similarity is the highest
                go2_ctrl.base_vel_cmd_input[0] = torch.tensor([velocity_value, 0.0, 0.0], dtype=torch.float32)
                # go2_ctrl.base_vel_cmd_input[0] = torch.tensor([velocity_value, -velocity_value/10, 0.0], dtype=torch.float32)


            with open("/blue/prabhat/duminduaelamurem/wd/isaac_sim/isaac-go2-ros2/results/August/08-18/debug.txt", "a") as f:
                f.write(f"Left Cosine Value: {left_cosine}\n")
                f.write(f"Middle Cosine Value: {middle_cosine}\n")
                f.write(f"Right Cosine Value: {right_cosine}\n")
        
        # # Navigate towards the target
        # if cosine_similarity < threshold: 
        #     # navstate = NavigationState.EXPLORATION
        #     go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, -velocity_value/10], dtype=torch.float32)

        else:
            go2_ctrl.base_vel_cmd_input[0] = torch.tensor([velocity_value, 0.0, 0.0], dtype=torch.float32)
        

    elif navstate == NavigationState.STOP:
        # Slow down when approaching the target
        go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    with open("/blue/prabhat/duminduaelamurem/wd/isaac_sim/isaac-go2-ros2/results/August/08-18/debug.txt", "a") as f:
        f.write(f"{ts} - Cosine similarity: {cosine_similarity:.4f} - {prompt}\n")
        f.write(f"{cosine_similarity > threshold}\n")
        f.write(f"Navigation State: {navstate}\n")
        f.write(f"Velocity Command: {go2_ctrl.base_vel_cmd_input[0].numpy()}\n")

    return navstate

