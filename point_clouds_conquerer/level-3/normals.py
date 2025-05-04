import open3d as o3d
import numpy as np

def reflectivity_threshold(pcd, thresh=0.45):
    colors = np.asarray(pcd.colors)
    reflectivities = colors[:, 0]
    # Get the point coordinates
    points = np.asarray(pcd.points)
    # Create a mask of points that have reflectivity above the threshold
    mask = reflectivities > thresh

    # Filter points and reflectivities using the mask
    filtered_points = points[mask]
    filtered_reflectivities = reflectivities[mask]

    # Create a new point cloud with the filtered points
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    return filtered_point_cloud

def roi_filter(pcd, roi_min=(-20,-6,-2), roi_max=(20,6,0)):
    points = np.asarray(pcd.points)

    mask_roi = np.logical_and.reduce((
        points[:, 0] >= roi_min[0],
        points[:, 0] <= roi_max[0],
        points[:, 1] >= roi_min[1],
        points[:, 1] <= roi_max[1],
        points[:, 2] >= roi_min[2],
        points[:, 2] <= roi_max[2]
    ))

    roi_points = points[mask_roi]

    # Create a new point cloud with the filtered points
    roi_pcd = o3d.geometry.PointCloud()
    roi_pcd.points = o3d.utility.Vector3dVector(roi_points)
    return roi_pcd


pcd = o3d.io.read_point_cloud("../data/KITTI_PCD/0000000357.pcd")
colors = np.asarray(pcd.colors)
filtered_point_cloud = reflectivity_threshold(pcd, thresh=0.45)
roi_pcd = roi_filter(filtered_point_cloud)
roi_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
o3d.visualization.draw_geometries([pcd.paint_uniform_color((0.8,0.8,0.8)), roi_pcd], point_show_normal=True)


for i in range(100):
	if i%10 == 0:	
		print("XYZ AND NORMAL FOR THE POINT", str(i))
		print(roi_pcd.points[i])
		print(roi_pcd.normals[i])

def normal_filter(pcd):
	direction = np.array([0, 0, 1])  # Z direction
	angle_tolerance = np.deg2rad(10)  # Convert to radians
	angles = np.arccos(np.dot(np.asarray(pcd.normals), direction))
	indices = np.where(np.abs(angles) < angle_tolerance)[0]
	filtered_pcd = pcd.select_by_index(indices)
	return filtered_pcd

normal_pcd = normal_filter(roi_pcd)
o3d.visualization.draw_geometries([pcd.paint_uniform_color((0.8,0.8,0.8)), normal_pcd], point_show_normal=True)

