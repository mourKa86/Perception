import open3d as o3d
import numpy as np
import struct
import glob
import os

size_float = 4
list_pcd = []

file_to_open="../data/KITTI_BIN/0000000000.bin"
file_to_save="../data/OUTPUT/0000000000_r.pcd"

def bin_to_pcd(bin_file, pcd_file):
    # Load .bin data
    scan = np.fromfile(bin_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Set the PointCloud's points
    pcd.points = o3d.utility.Vector3dVector(scan[:,:3])

    # Set the PointCloud's intensity
    reflectance = #TODO: Get the reflectance and color
    colors = #TODO: Get the reflectance and color
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # Save pcd file
    o3d.io.write_point_cloud(pcd_file, pcd)

bin_to_pcd(file_to_open, file_to_save)
pcd = o3d.io.read_point_cloud(file_to_save)
o3d.visualization.draw_geometries([pcd]) 
