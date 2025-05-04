import numpy as np
import open3d as o3d
import struct
import struct
import os

bin_file = "data\\KITTI_BIN\\0000000000.bin"  # Replace with your .bin file
pcd_file = "data\\KITTI_BIN\\output.pcd" 


def bin_to_pcd(bin_file, pcd_file):
    # Load .bin data
    scan = np.fromfile(bin_file, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()

    # Set the PointCloud's points
    pcd.points = o3d.utility.Vector3dVector(scan[:,:3])
    reflectance = scan[:,3]
    colors = np.zeros((reflectance.shape[0], 3))
    colors[:,0] = reflectance
    pcd.colors = o3d.utility.Vector3dVector(colors) 


    # Save pcd file
    o3d.io.write_point_cloud(pcd_file, pcd)


bin_to_pcd(bin_file, pcd_file)

pcd = o3d.io.read_point_cloud(pcd_file)
o3d.visualization.draw_geometries([pcd])