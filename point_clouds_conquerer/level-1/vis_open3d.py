import open3d as o3d
import struct

#pcd = o3d.io.read_point_cloud("../data/PLAYGROUND/car.ply")
#pcd = o3d.io.read_point_cloud("../data/PLAYGROUND/bunny.pcd")
#pcd = o3d.io.read_point_cloud("../data/APOLLO/bg_1534313592.018882.pcd.ply")

#pcd = o3d.io.read_point_cloud("../data/UDACITY/000000.pcd")
#pcd = o3d.io.read_point_cloud("../data/PLAYGROUND/turtlebot-pointcloud 2.ply")
# pcd = o3d.io.read_point_cloud("data\\KITTI_PCD\\000000.pcd")

pcd = o3d.io.read_point_cloud("data\\KITTI_BIN\\output.pcd")



o3d.visualization.draw_geometries([pcd]) 

