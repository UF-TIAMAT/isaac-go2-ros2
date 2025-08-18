# with open("/blue/prabhat/duminduaelamurem/wd/isaac_sim/isaac-go2-ros2/cfg/debug.yaml", ) as f:
#     debug_cfg = yaml.safe_load(f)

# if rgb.shape == (480, 640, 3):

#     PROMPT = debug_cfg["prompt"]

#     cosine_similarity = similarity_model.cosine_image_text(cameras[0].get_rgb(), PROMPT)
#     print(f"\rCosine similarity: {cosine_similarity:.4f} - {PROMPT}", end='', flush=True)


# if cosine_similarity > threashold:
# # Move forward
# go2_ctrl.base_vel_cmd_input[0] = torch.tensor([lin_vel, 0, 0], dtype=torch.float32)

# if debug_cfg["debug"]:
# step_count += 1

# if step_count < max_steps:
#     ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     # Image.fromarray(rgb).save(f"/blue/prabhat/duminduaelamurem/wd/isaac_sim/isaac-go2-ros2/results/August/08-18/rgb/frame_{ts}.png")
#     with open("/blue/prabhat/duminduaelamurem/wd/isaac_sim/isaac-go2-ros2/results/August/08-18/debug.txt", "a") as f:
#         f.write(f"{ts} - Cosine similarity: {cosine_similarity:.4f} - {PROMPT}\n")
# else:
# step_count = 0


# # NOTE: Temporary, save observations
# import pickle
# save_path = '/blue/prabhat/duminduaelamurem/wd/isaac_sim/isaac-go2-ros2/results/August/08-17/'
# if not os.path.exists(save_path):
#     os.makedirs(save_path)
# with open(f'{save_path}/rgb/rgb_{step_count}.pkl', 'wb') as f:
#     pickle.dump(cameras[0].get_rgb(), f)

# with open(f'{save_path}/depth/depth_{step_count}.pkl', 'wb') as f:
#     pickle.dump(cameras[0].get_depth(), f)

# step_count += 1
