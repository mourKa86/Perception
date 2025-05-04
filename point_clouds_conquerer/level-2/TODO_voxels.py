import open3d as o3d 
import numpy as np


pcd = o3d.io.read_point_cloud("../data/PLAYGROUND/turtlebot-pointcloud 2.ply")
#o3d.visualization.draw_geometries([pcd])

### DOWNSAMPLING
print(f"Points before downsampling: {len(pcd.points)} ")
# pcd =
print(f"Points after downsampling: {len(pcd.points)}")
# o3d.visualization.draw_geometries([pcd])

### VOXELIZATION
voxel_size = 0.01
# voxel_grid=
# o3d.visualization.draw_geometries([voxel_grid])
