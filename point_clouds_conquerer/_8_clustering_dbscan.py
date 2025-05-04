import numpy as np
import open3d as o3d
import glob
from matplotlib import pyplot as plt
from open3d.visualization.draw_plotly import get_plotly_fig


def roi_filter(pcd, roi_min=(-30, -10, -3), roi_max=(30, 10, 2)):
    points = np.asarray(pcd.points)

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = None  # Handle missing colors

    mask_roi = np.logical_and.reduce((
        points[:, 0] >= roi_min[0],
        points[:, 0] <= roi_max[0],
        points[:, 1] >= roi_min[1],
        points[:, 1] <= roi_max[1],
        points[:, 2] >= roi_min[2],
        points[:, 2] <= roi_max[2]
    ))

    roi_points = points[mask_roi]

    # Handle missing colors
    if colors is not None:
        roi_colors = colors[mask_roi]
    else:
        roi_colors = None  # No colors

    # Create a new point cloud with the filtered points
    roi_pcd = o3d.geometry.PointCloud()
    roi_pcd.points = o3d.utility.Vector3dVector(roi_points)

    if roi_colors is not None:
        roi_pcd.colors = o3d.utility.Vector3dVector(roi_colors)

    return roi_pcd

def ransac(pcd, distance_threshold=0.33, ransac_n=3, num_iterations=100):
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0.5, 0.75, 0.25])
    inlier_cloud.paint_uniform_color([0.25, 0.5, 0.75])
    return inlier_cloud, outlier_cloud

def dbscan(outlier_cloud, eps=1.5, min_points=10):
    outlier_cloud = roi_filter(outlier_cloud, roi_min=(-20, -8, -2), roi_max=(20, 8, 0))
    labels = np.array(outlier_cloud.cluster_dbscan(eps=eps, min_points=min_points))
    print("Unique DBSCAN Labels:", np.unique(labels))
    max_label = labels.max()
    if max_label == -1:
        print("No clusters found, returning unchanged cloud.")
        return outlier_cloud

    colors = plt.get_cmap("inferno_r")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return outlier_cloud

if __name__ == "__main__":
    point_cloud_files = sorted(glob.glob("KITTI_PCD/*.pcd"))
    pcd = o3d.io.read_point_cloud(point_cloud_files[200])
    inlier_cloud, outlier_cloud = ransac(pcd, distance_threshold=0.33, ransac_n=3, num_iterations=100)
    roi_outlier_cloud = dbscan(outlier_cloud, eps=1.5, min_points=10)

    o3d.visualization.draw_geometries([inlier_cloud, roi_outlier_cloud])