# basic libraries imports
import sys
import numpy as np
import random

# deep learning libraries
import torch
import cumm.tensorview as tv  # CUDA Matrix multiplication library
import spconv   # Sparse convolution library
from spconv.utils import Point2VoxelCPU3d as VoxelGenerator


# plotting library
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# custom libraries imports
from data.lidar_od_scripts.gpuVersion.gpuVersion.visual_utils import plot_pc_data3d,PCD_SCENE, PCD_CAM_VIEW, boxes_to_corners_3d, plot_bboxes_3d



sys.path.append('data/lidar_od_scripts/gpuVersion/gpuVersion/')

import pickle
with open('data/lidar_od_scripts/AxisAlignedTargetAssigner_GPU.pkl', 'rb') as f:
    data = pickle.load(f)

# RGB image visualization
points = data['points']
image = data['images']
# print(f"points.shape = {points.shape}")
# print(f"image.shape = {image.shape}")


# ----- Plot Image ----- #
# plt.figure(figsize=(12,6))
# plt.imshow(image)
# plt.axis('off')
# plt.show()

# ----- Plot Point Cloud ----- #
# lidar_3d_plots = [plot_pc_data3d(x=points[:,0], y=points[:,1], z=points[:,2], colorscale='viridis')]
# layout = dict(template="plotly_dark", scene_camera = PCD_CAM_VIEW, scene = PCD_SCENE, title="POINT CLOUD VISUALIZATION")
# fig = go.Figure(data=lidar_3d_plots, layout=layout)
# fig.show()

# ----- Voxelization in GPU ----- #
vsize_xyz = np.array([0.5, 0.5, 0.5])                  # voxel size in x,y,z
coors_range_xyz = np.array([0, -40, -3, 70.4, 40, 1])  # KITTI point cloud range
num_point_features = 4                                 # x,y,z, intensity
max_num_points_per_voxel = 200
max_num_voxels = 200

voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

# generate voxels from point cloud data
tv_voxels, tv_voxel_indices, tv_num_points = voxel_generator.point_to_voxel(tv.from_numpy(points))

voxels = tv_voxels.numpy()
voxel_indices = tv_voxel_indices.numpy()
num_points = tv_num_points.numpy()

# print(f"voxels.shape = {voxels.shape}")
# print(f"voxel_indices.shape = {voxel_indices.shape}")
# print(f"num_points.shape = {num_points.shape}")

# ----- calculating voxel centers and corners ----- #
# converting grid indices to actual 3D coordinates
# indices give bottom left corner, add half voxel size to get voxel centres
voxelCentres = (voxel_indices[:, ::-1] * vsize_xyz) + (coors_range_xyz[0:3]) + (vsize_xyz * 0.5)

# Changing to [x,y,z,dx,dy,dz,yaw] format to find voxel corners
voxelBBoxes = np.column_stack((voxelCentres, np.repeat( np.append(vsize_xyz, 0.0)[None,:], len(voxel_indices), axis=0)))
print(f"voxelBBoxes.shape = {voxelBBoxes.shape}")

voxelCorners = boxes_to_corners_3d(voxelBBoxes)
print(f"voxelCorners.shape = {voxelCorners.shape}")

# ----- Plot Voxel Generator ----- #
hexadecimal_alphabets = '0123456789ABCDEF'
color = ["#" + ''.join([random.choice(hexadecimal_alphabets) for j in range(6)]) for i in range(voxel_indices.shape[0])]

voxel_generator_plots = [
    plot_pc_data3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], opacity=0.1, colorscale='viridis')]
voxel_generator_plots.extend(plot_bboxes_3d(voxelCorners, color))

for vox_idx in range(voxel_indices.shape[0]):
    voxel_points = voxels[vox_idx, 0: num_points[vox_idx], 0:3]
    voxel_generator_plots.append(plot_pc_data3d(x=voxel_points[:, 0], y=voxel_points[:, 1], z=voxel_points[:, 2],
                                                apply_color_gradient=False, color=color[vox_idx]))

layout = dict(scene=PCD_SCENE, scene_camera=PCD_CAM_VIEW, title="VOXEL GENERATOR PLOTS", template="plotly_dark")
fig = go.Figure(data=voxel_generator_plots, layout=layout)
fig.show()

# ----- Plotting Voxels ----- #
plt.figure(figsize=(8,8))
plt.scatter(points[:, 1], points[:, 0], alpha=0.1, s=2)

for vox_idx in range(len(voxel_indices)):
    voxel_points = voxels[vox_idx, 0 : num_points[vox_idx], 0:3]
    plt.scatter(voxel_points[:,1], voxel_points[:,0], s=2, label=f"Voxel_{vox_idx}")
    voxelBoxCornersBEV = voxelCorners[vox_idx, [0,1,2,3,0], 0:2]
    plt.plot(voxelBoxCornersBEV[:,1], voxelBoxCornersBEV[:,0], label=f"Voxel_BBox{vox_idx}")

plt.legend()
plt.show()