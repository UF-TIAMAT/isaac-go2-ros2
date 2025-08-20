# import time 

# import open3d as o3d
# import open3d.core as o3c
# from tqdm import tqdm 

# device = o3c.Device("CUDA:0")

# vbg = o3d.t.geometry.VoxelBlockGrid(
#          attr_names=("tsdf", "weight"),
#          attr_dtypes=(o3c.float32, o3c.float32),
#          voxel_size=0.3/512, 
#          block_resolution=16,
#          block_count=50000,
#          device=device)

# start = time.time()

# n_files = 10

# for i in tqdm(range(n_files)):
#     depth = o3d.t.io.read_image(f"data/depth_{i}.png").to(device)
#     extrinsic = extrinisics[i]

#     frustrum_block_coords = vbg.compute_unique_block_coordinates(
#         depth, depth_intrinsic, extrinsic, config.depth_scale, config.depth_max
#     )

#     vbg.integrate(frustrum_block_coords, depth, extrinsic, depth_intrinsic, config.depth_scale, config.depth_max)

#     dt = time.time() - start

    
