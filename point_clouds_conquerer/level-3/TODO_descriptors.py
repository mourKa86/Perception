import open3d as o3d 
import numpy as np

pcd = o3d.io.read_point_cloud("../data/PLAYGROUND/car.ply")
#TODO: Estimate normals
points = np.asarray(pcd.points)

print("NUMBER OF POINTS:")
print(len(points))
# Compute FPFH descriptors for each point
radius_feature = 0.1  # Adjust the feature radius as needed
fpfh = 

print('WHAT WE GET OUT OF FPFH:')
print(np.asarray(fpfh.data).shape)
print("POINT INDEX 0")
print(np.asarray(pcd.points)[0])
print("FEATURE VECTOR FOR THE POINT")
print(np.asarray(fpfh.data[:,0]))

variances = np.var(fpfh.data, axis=1)
top_dims = np.argpartition(variances, -3)[-3:]
fpfh = fpfh.data.T[top_dims].T

# Normalize the FPFH descriptors to [0, 1]
fpfh_normalized = (fpfh.data - np.min(fpfh.data)) / (np.max(fpfh.data) - np.min(fpfh.data))

# Assign the FPFH descriptors as colors to the point cloud
pcd.colors = o3d.utility.Vector3dVector(fpfh_normalized)

# Visualize the point cloud with color-coded descriptors
o3d.visualization.draw_geometries([pcd])
