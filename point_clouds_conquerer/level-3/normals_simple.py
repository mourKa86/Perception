import open3d as o3d
import numpy as np

#pcd = o3d.io.read_point_cloud("car.ply")
pcd = o3d.io.read_point_cloud("../data/PLAYGROUND/car.ply")

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))

o3d.visualization.draw_geometries([pcd], point_show_normal=True)