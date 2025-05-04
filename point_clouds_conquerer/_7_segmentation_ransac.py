import numpy as np
import open3d as o3d
import glob

def ransac(pcd, distance_threshold=0.33, ransac_n=3, num_iterations=100):
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0.5, 0.75, 0.25])
    inlier_cloud.paint_uniform_color([0.25, 0.5, 0.75])
    return inlier_cloud, outlier_cloud

if __name__ == "__main__":
    point_cloud_files = sorted(glob.glob("KITTI_PCD/*.pcd"))
    pcd = o3d.io.read_point_cloud(point_cloud_files[200])
    inlier_cloud, outlier_cloud = ransac(pcd, distance_threshold=0.33, ransac_n=3, num_iterations=100)


    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


