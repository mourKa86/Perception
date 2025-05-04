import open3d as o3d 
import numpy as np

#pcd = o3d.io.read_point_cloud("../data/PLAYGROUND/car.ply")
#pcd = o3d.io.read_point_cloud("../data/PLAYGROUND/bunny.pcd")
#pcd = o3d.io.read_point_cloud("../data/APOLLO/bg_1534313592.018882.pcd.ply")

pcd = o3d.io.read_point_cloud("../data/UDACITY/000000.pcd")
#pcd = o3d.io.read_point_cloud("../data/PLAYGROUND/turtlebot-pointcloud 2.ply")
#pcd = o3d.io.read_point_cloud("../data/KITTI_PCD/0000000200.pcd")

def ransac(pcd, distance_threshold=0.33, ransac_n=3, num_iterations=100):
    return inlier_cloud, outlier_cloud

inlier_cloud, outlier_cloud = ransac(pcd, distance_threshold=0.33, ransac_n=3, num_iterations=100)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

