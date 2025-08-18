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
    

def vlfm_navigation(rgb: np.ndarray, depth: np.ndarray, prompt: str, navstate: NavigationState, similarity_model: HFBLIP2ImageTextRetrieval, threshold: float = 0.5) -> NavigationState:
    
    cosine_similarity = similarity_model.cosine_image_text(rgb, prompt)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")



    velocity_value = 0.25  # Default forward velocity/ angular velocity


    if navstate == NavigationState.EXPLORATION:

        if cosine_similarity > threshold:
            # Move forward in exploration state
            go2_ctrl.base_vel_cmd_input[0] = torch.tensor([velocity_value, 0.0, 0.0], dtype=torch.float32)
            navstate = NavigationState.NAVIGATION
        else:
            # Continue exploring by turning in place
            go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, velocity_value], dtype=torch.float32)

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
            elif right_cosine > left_cosine and right_cosine > middle_cosine:
                # Turn right
                go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, -velocity_value], dtype=torch.float32)
            else:
                # Move forward if middle cosine similarity is the highest
                go2_ctrl.base_vel_cmd_input[0] = torch.tensor([velocity_value, 0.0, 0.0], dtype=torch.float32)

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

