import os
import hydra
import rclpy
import torch
import time
import math
import argparse
import yaml
from PIL import Image
from datetime import datetime

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from go2.go2_env import Go2RSLEnvCfg, camera_follow
import env.sim_env as sim_env
import go2.go2_sensors as go2_sensors
import omni
import carb
import go2.go2_ctrl as go2_ctrl
import ros2.go2_ros2_bridge as go2_ros2_bridge
from transfer.navigation import vlfm_navigation, NavigationState
from transfer.cosine_similarity import HFBLIP2ImageTextRetrieval

FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")


@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):

    # Go2 Environment setup
    go2_env_cfg = Go2RSLEnvCfg()
    go2_env_cfg.scene.num_envs = cfg.num_envs
    go2_env_cfg.decimation = math.ceil(1./go2_env_cfg.sim.dt/cfg.freq)
    go2_env_cfg.sim.render_interval = go2_env_cfg.decimation
    go2_ctrl.init_base_vel_cmd(cfg.num_envs)
    # env, policy = go2_ctrl.get_rsl_flat_policy(go2_env_cfg)
    env, policy = go2_ctrl.get_rsl_rough_policy(go2_env_cfg)

    similarity_model = HFBLIP2ImageTextRetrieval()

    # Simulation environment
    if (cfg.env_name == "obstacle-dense"):
        sim_env.create_obstacle_dense_env() # obstacles dense
    elif (cfg.env_name == "obstacle-medium"):
        sim_env.create_obstacle_medium_env() # obstacles medium
    elif (cfg.env_name == "obstacle-sparse"):
        sim_env.create_obstacle_sparse_env() # obstacles sparse
    elif (cfg.env_name == "warehouse"):
        sim_env.create_warehouse_env() # warehouse
    elif (cfg.env_name == "warehouse-forklifts"):
        sim_env.create_warehouse_forklifts_env() # warehouse forklifts
    elif (cfg.env_name == "warehouse-shelves"):
        sim_env.create_warehouse_shelves_env() # warehouse shelves
    elif (cfg.env_name == "full-warehouse"):
        sim_env.create_full_warehouse_env() # full warehouse

    # Sensor setup
    sm = go2_sensors.SensorManager(cfg.num_envs)
    lidar_annotators = sm.add_rtx_lidar()
    cameras = sm.add_camera(cfg.freq)

    # Keyboard control
    system_input = carb.input.acquire_input_interface()
    system_input.subscribe_to_keyboard_events(
        omni.appwindow.get_default_app_window().get_keyboard(), go2_ctrl.sub_keyboard_event)
    
    # ROS2 Bridge
    rclpy.init()
    dm = go2_ros2_bridge.RobotDataManager(env, lidar_annotators, cameras, cfg)

    # Run simulation
    sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)
    obs, _ = env.reset()
    navigation_state = NavigationState.EXPLORATION

    # # Limit num steps
    # max_steps = 100
    # step_count = 0

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging_file = f"/blue/prabhat/duminduaelamurem/wd/isaac_sim/isaac-go2-ros2/results/August/08-21/{ts}.txt"


    while simulation_app.is_running():
        start_time = time.time()

        # if step_count >= max_steps:
        #     break

        with torch.inference_mode():            
            # control joints
            actions = policy(obs)

            # step the environment
            obs, _, _, _ = env.step(actions)

            threashold = 0.53
            lin_vel = 1.5 
            step_count = 0  
            max_steps = 100

            with open("/blue/prabhat/duminduaelamurem/wd/isaac_sim/isaac-go2-ros2/cfg/debug.yaml", ) as f:
                            debug_cfg = yaml.safe_load(f)

            # Get RGB and Depth data
            rgb = cameras[0].get_rgb()
            depth = cameras[0].get_depth()
            prompt = debug_cfg["prompt"]   


            # Get robot odometry 
            robot_data = env.unwrapped.scene["unitree_go2"].data
            odometry = {
                 "position": robot_data.root_state_w[0, :3], #x, y, z (torch.tensor format)
                 "rotation": robot_data.root_state_w[0, 3:7], #x, y, z, w (torch.tensor format)
                 "linear_velocity": robot_data.root_lin_vel_b[0], #x, y, z (torch.tensor format)
                 "angular_velocity": robot_data.root_ang_vel_b[0] #x, y, z (torch.tensor format)
            }



            if rgb.shape == (480, 640, 3):
                 navigation_state = vlfm_navigation(rgb, depth, prompt, navigation_state, similarity_model, threashold, odometry, logging_file)
                

            # NOTE: This for debugging purposes only 
        
            # if rgb.shape == (480, 640, 3):

            #     PROMPT = debug_cfg["prompt"]

            #     cosine_similarity = similarity_model.cosine_image_text(cameras[0].get_rgb(), PROMPT)
            #     go2_ctrl.base_vel_cmd_input[0] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)

            #     print(f"\rCosine similarity: {cosine_similarity:.4f} - {PROMPT}", end='', flush=True)

            #     if cosine_similarity > threashold:
            #         # Move forward
            #         pass

            #     if debug_cfg["debug"]:
            #         step_count += 1

            #         if step_count < max_steps:
            #             ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            #             # Image.fromarray(rgb).save(f"/blue/prabhat/duminduaelamurem/wd/isaac_sim/isaac-go2-ros2/results/August/08-18/rgb/frame_{ts}.png")
            #             with open("/blue/prabhat/duminduaelamurem/wd/isaac_sim/isaac-go2-ros2/results/August/08-18/debug.txt", "a") as f:
            #                 f.write(f"{ts} - Cosine similarity: {cosine_similarity:.4f} - {PROMPT}\n")
            #     else:
            #         step_count = 0
            





            # # ROS2 data
            dm.pub_ros2_data()
            rclpy.spin_once(dm)

            # Camera follow
            if (cfg.camera_follow):
                camera_follow(env)

            # limit loop time
            elapsed_time = time.time() - start_time
            if elapsed_time < sim_step_dt:
                sleep_duration = sim_step_dt - elapsed_time
                time.sleep(sleep_duration)
        actual_loop_time = time.time() - start_time
        rtf = min(1.0, sim_step_dt/elapsed_time)
        print(f"\rStep time: {actual_loop_time*1000:.2f}ms, Real Time Factor: {rtf:.2f}", end='', flush=True)
    
    dm.destroy_node()
    rclpy.shutdown()
    simulation_app.close()

if __name__ == "__main__":
    run_simulator()
    