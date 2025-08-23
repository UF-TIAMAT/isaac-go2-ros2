
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
    COLLISION_AVOIDANCE = 4  # New stecho ate for collision avoidance
    

def check_collision_risk(depth: np.ndarray, min_distance: float = 1.1, critical_distance: float = 0.9, logging_file: str = "temp.txt") -> tuple:
    """
    Check for collision risk using depth information
    
    Args:
        depth: Depth image array
        min_distance: Distance threshold for collision warning (meters)
        critical_distance: Distance threshold for critical collision risk (meters)
    
    Returns:
        tuple: (has_collision_risk, critical_collision, collision_direction)
               collision_direction: 'left', 'center', 'right', or 'none'
    """
    if depth is None or depth.size == 0:
        return False, False, 'none'
    
    # Remove invalid depth values (0 or inf)
    valid_depth = depth[depth > 0]
    if valid_depth.size == 0:
        return False, False, 'none'
    
    # Split depth image into regions
    height, width = depth.shape
    center_region = depth[:, width//3:2*width//3]
    left_region = depth[:, :width//3]
    right_region = depth[:, 2*width//3:]
    
    # Focus on lower portion of image (ground level obstacles)
    focus_height_low = int(height * 0.4)  # Lower 40% of image
    focus_height_upper = int(height * 0.5)
    center_focus = center_region[focus_height_low:focus_height_upper, :]
    left_focus = left_region[focus_height_low:focus_height_upper, :]
    right_focus = right_region[focus_height_low:focus_height_upper, :]

    # with open(logging_file, "a") as f:
    #     for i in range(32):
    #         strip_start = height//32 * i
    #         strip_end = height//32 * (i + 1)
    #         center_strip = center_region[strip_start:strip_end, :]
    #         left_strip = left_region[strip_start:strip_end, :]
    #         right_strip = right_region[strip_start:strip_end, :]
    #         center_min = np.min(center_strip[center_strip > 0]) if np.any(center_strip > 0) else float('inf')
    #         left_min = np.min(left_strip[left_strip > 0]) if np.any(left_strip > 0) else float('inf')
    #         right_min = np.min(right_strip[right_strip > 0]) if np.any(right_strip > 0) else float('inf')
    #         f.write(f"Strip {i}: Center: {center_min:.2f}, Left: {left_min:.2f}, Right: {right_min:.2f}\n")

    
    # Calculate minimum distances in each region
    center_min = np.min(center_focus[center_focus > 0]) if np.any(center_focus > 0) else float('inf')
    left_min = np.min(left_focus[left_focus > 0]) if np.any(left_focus > 0) else float('inf')
    right_min = np.min(right_focus[right_focus > 0]) if np.any(right_focus > 0) else float('inf')
    
    # Check for collision risk
    has_collision_risk = center_min < min_distance or left_min < min_distance or right_min < min_distance
    critical_collision = center_min < critical_distance or left_min < critical_distance or right_min < critical_distance
    
    # Determine safest direction
    collision_direction = 'none'
    if has_collision_risk:
        if left_min > right_min and left_min > center_min:
            collision_direction = 'left'
        elif right_min > left_min and right_min > center_min:
            collision_direction = 'right'
        else:
            collision_direction = 'center'

    # Log collision risk
    with open(logging_file, "a") as f:
        f.write(f"Collision Risk - Center: {center_min:.2f}, Left: {left_min:.2f}, Right: {right_min:.2f}\n")
        f.write(f"Has Collision Risk: {has_collision_risk}, Critical: {critical_collision}, Direction: {collision_direction}\n")
    
    return has_collision_risk, critical_collision, collision_direction


def get_safe_velocity(navstate: NavigationState, collision_risk: bool, critical_collision: bool, 
                     collision_direction: str, base_velocity: float = 0.25) -> torch.Tensor:
    """
    Calculate safe velocity command based on collision risk
    
    Args:
        navstate: Current navigation state
        collision_risk: Whether there's a collision risk
        critical_collision: Whether there's critical collision risk
        collision_direction: Direction of safest path ('left', 'right', 'center', 'none')
        base_velocity: Base velocity value
    
    Returns:
        torch.Tensor: Velocity command [linear_x, linear_y, angular_z]
    """
    if critical_collision:
        # Emergency stop
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    
    if collision_risk:
        # Reduce speed and prefer turning
        reduced_velocity = base_velocity * 0.3
        if collision_direction == 'left':
            return torch.tensor([reduced_velocity * 0.5, 0.0, -base_velocity * 0.8], dtype=torch.float32)
        elif collision_direction == 'right':
            return torch.tensor([reduced_velocity * 0.5, 0.0, base_velocity * 0.8], dtype=torch.float32)
        else:
            return torch.tensor([reduced_velocity, 0.0, 0.0], dtype=torch.float32)
    
    # No collision risk - return normal velocity
    return torch.tensor([base_velocity, 0.0, 0.0], dtype=torch.float32)


def vlfm_navigation(rgb: np.ndarray, depth: np.ndarray, prompt: str, navstate: NavigationState, 
                   similarity_model: HFBLIP2ImageTextRetrieval, threshold: float = 0.32, 
                   odometry: dict = None, logging_file: str ="temp.txt", collision_distance: float = 1.1, 
                   critical_distance: float = 0.9) -> NavigationState:

    cosine_similarity = similarity_model.cosine_image_text(rgb, prompt)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    # Collision detection
    collision_risk, critical_collision, collision_direction = check_collision_risk(
        depth, collision_distance, critical_distance, logging_file=logging_file
    )

    velocity_value = 0.25  # Default forward velocity/ angular velocity
    min_threshold = 0.25

    with open(logging_file, "a") as f:
        f.write(f"{ts} - Cosine similarity: {cosine_similarity:.4f} - {prompt}\n")
        f.write(f"Similarity above threshold: {cosine_similarity > threshold}\n")
        f.write(f"Navigation State: {navstate}\n")
        f.write(f"Collision Risk: {collision_risk}, Critical: {critical_collision}, Direction: {collision_direction}\n")

    # Handle critical collision first (emergency stop)
    if critical_collision:
        go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        navstate = NavigationState.COLLISION_AVOIDANCE
        
    elif navstate == NavigationState.EXPLORATION:
        if cosine_similarity > threshold and not collision_risk:
            # Move forward in exploration state if no collision risk
            go2_ctrl.base_vel_cmd_input[0] = get_safe_velocity(
                navstate, collision_risk, critical_collision, collision_direction, velocity_value
            )
            navstate = NavigationState.NAVIGATION
        else:
            # Continue exploring by turning in place, avoiding obstacles
            if collision_risk:
                if collision_direction == 'left':
                    go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, -velocity_value], dtype=torch.float32)
                elif collision_direction == 'right':
                    go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, velocity_value], dtype=torch.float32)
                else:
                    go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, velocity_value], dtype=torch.float32)
            else:
                go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, velocity_value], dtype=torch.float32)

    elif navstate == NavigationState.NAVIGATION:
        left = rgb[:, :int(rgb.shape[1] / 3)]
        middle = rgb[:, int(rgb.shape[1] / 3):int(2 * rgb.shape[1] / 3)]
        right = rgb[:, int(2 * rgb.shape[1] / 3):]

        # Check if the cosine similarity is below the threshold
        # if cosine_similarity < threshold:
        left_cosine = similarity_model.cosine_image_text(left, prompt)
        middle_cosine = similarity_model.cosine_image_text(middle, prompt)
        right_cosine = similarity_model.cosine_image_text(right, prompt)


        if collision_risk:
            navstate = NavigationState.STOP
        # if collision_risk:

        #     if cosine_similarity > threshold:
        #         go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        #         navstate = NavigationState.STOP

        #     else:
        #         go2_ctrl.base_vel_cmd_input[0] = get_safe_velocity(
        #             navstate, collision_risk, critical_collision, collision_direction, velocity_value
        #         )
        #         navstate = NavigationState.COLLISION_AVOIDANCE

        else:

            if right_cosine >= left_cosine and right_cosine >= middle_cosine:
                go2_ctrl.base_vel_cmd_input[0] = torch.tensor([velocity_value, 0.0, -velocity_value], dtype=torch.float32)
            elif left_cosine >= right_cosine and left_cosine >= middle_cosine:
                go2_ctrl.base_vel_cmd_input[0] = torch.tensor([velocity_value, 0.0, velocity_value], dtype=torch.float32)
            else:
                go2_ctrl.base_vel_cmd_input[0] = torch.tensor([velocity_value, 0.0, 0.0], dtype=torch.float32)

            navstate = NavigationState.NAVIGATION

            if cosine_similarity < min_threshold:
                go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
                navstate = NavigationState.EXPLORATION
                

        with open(logging_file, "a") as f:
            f.write(f"Left Cosine Value: {left_cosine}\n")
            f.write(f"Middle Cosine Value: {middle_cosine}\n")
            f.write(f"Right Cosine Value: {right_cosine}\n")


        # if cosine_similarity > threshold:
        #     # Determine the direction to turn based on cosine similarity and collision avoidance
        #     if collision_risk:
        #         # Priority: avoid collision while considering target direction
        #         if collision_direction == 'left' and right_cosine >= left_cosine:
        #             # Turn right (away from obstacle)
        #             go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, -velocity_value * 0.8], dtype=torch.float32)
        #         elif collision_direction == 'right' and left_cosine >= right_cosine:
        #             # Turn left (away from obstacle)
        #             go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, velocity_value * 0.8], dtype=torch.float32)
        #         else:
        #             # General collision avoidance
        #             if collision_direction == 'left':
        #                 go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, -velocity_value], dtype=torch.float32)
        #             elif collision_direction == 'right':
        #                 go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, velocity_value], dtype=torch.float32)
        #             else:
        #                 go2_ctrl.base_vel_cmd_input[0] = torch.tensor([velocity_value * 0.3, 0.0, 0.0], dtype=torch.float32)
        #         navstate = NavigationState.COLLISION_AVOIDANCE
        #     else:
        #         # Original logic when no collision risk
        #         if left_cosine > right_cosine and left_cosine > middle_cosine:
        #             # Turn left
        #             go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, velocity_value], dtype=torch.float32)
        #         elif right_cosine > left_cosine and right_cosine > middle_cosine:
        #             # Turn right
        #             go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, -velocity_value], dtype=torch.float32)
        #         elif middle_cosine > threshold:
        #             # Move forward if middle cosine similarity is the highest
        #             go2_ctrl.base_vel_cmd_input[0] = torch.tensor([velocity_value, 0.0, 0.0], dtype=torch.float32)



        # else:
        #     # Navigate towards the target with collision avoidance
        #     if collision_risk:
        #         go2_ctrl.base_vel_cmd_input[0] = get_safe_velocity(
        #             navstate, collision_risk, critical_collision, collision_direction, velocity_value
        #         )
        #         navstate = NavigationState.COLLISION_AVOIDANCE
        #     else:
        #         go2_ctrl.base_vel_cmd_input[0] = torch.tensor([velocity_value, 0.0, 0.0], dtype=torch.float32)

    elif navstate == NavigationState.COLLISION_AVOIDANCE:
        # In collision avoidance state, prioritize safety
        if not collision_risk:
            # No more collision risk, return to appropriate state
            if cosine_similarity > threshold:
                navstate = NavigationState.NAVIGATION
            else:
                navstate = NavigationState.EXPLORATION
        else:
            # Continue collision avoidance
            if collision_direction == 'left':
                go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, -velocity_value * 0.7], dtype=torch.float32)
            elif collision_direction == 'right':
                go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, velocity_value * 0.7], dtype=torch.float32)
            else:
                go2_ctrl.base_vel_cmd_input[0] = torch.tensor([velocity_value * 0.2, 0.0, 0.0], dtype=torch.float32)

    elif navstate == NavigationState.STOP:
        # Slow down when approaching the target
        go2_ctrl.base_vel_cmd_input[0] = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

    # Enhanced logging
    with open(logging_file, "a") as f:
        f.write(f"Velocity Command: {go2_ctrl.base_vel_cmd_input[0].numpy()}\n")
        f.write(f"Navigation State: {navstate}")
        f.write("-" * 50 + "\n")

    return navstate