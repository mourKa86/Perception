import open3d as o3d
import numpy as np

#pcd = o3d.io.read_point_cloud("../data/PLAYGROUND/car.ply")
pcd = o3d.io.read_point_cloud("../data/KITTI_PCD/0000000000.pcd")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

#o3d.visualization.draw_geometries([pcd], point_show_normal=True)

### 1. ALPHA SHAPES
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, 0.037)
mesh.compute_vertex_normals()
#o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True)

### 2. BALL PIVOTING
radii = [0.08, 0.16, 0.24, 0.4]
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
mesh.compute_vertex_normals()
#o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True)


# 3. POISSON
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, scale = 0.4, linear_fit=True)  # Convert point cloud to mesh
mesh.filter_smooth_laplacian(2)
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([pcd, mesh], mesh_show_back_face=True)

