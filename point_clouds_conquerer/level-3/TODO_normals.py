import open3d as o3d
import numpy as np

def reflectivity_threshold(pcd, thresh=0.45):
    pass

def roi_filter(pcd, roi_min=(-20,-6,-2), roi_max=(20,6,0)):
    pass

#TODO: CALL YOUR PIPELINE, BUT ADD NORMALS ESTIMATION
pcd = o3d.io.read_point_cloud("../data/KITTI_PCD/0000000357.pcd")


def normal_filter(pcd):
    #TODO: Filter our normals that are horizontal
    pass

normal_pcd = normal_filter(roi_pcd)
o3d.visualization.draw_geometries([pcd.paint_uniform_color((0.8,0.8,0.8)), normal_pcd], point_show_normal=True)

