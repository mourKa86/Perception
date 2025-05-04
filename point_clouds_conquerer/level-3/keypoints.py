import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA

pcd = o3d.io.read_point_cloud("../data/PLAYGROUND/car.ply")
pcd.paint_uniform_color((0.5,0.5,0.5))

### 1. RANDOM SELECTION
#keypoint_indices = np.random.choice(len(pcd.points), size=100, replace=False)
#o3d.visualization.draw_geometries([pcd, pcd.select_by_index(keypoint_indices).paint_uniform_color((1,0,0))])


### 2. INTRINSIC SHAPE SIGNATURE
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
keypoints.paint_uniform_color([1.0, 0.0, 0.0])
o3d.visualization.draw_geometries([pcd,keypoints])


### 3. FIT A LINE
pca = PCA(n_components=2)
pca.fit(np.asarray(keypoints.points))

# The line is along the PC1, which is the eigenvector corresponding to the largest eigenvalue
line_vector = pca.components_[0]

# Compute two points along the line: one at the mean (center of points), one at some distance away
center = pca.mean_
point1 = center + line_vector
point2 = center - line_vector

# Create a LineSet from the two points
lines = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector([point1, point2]),
    lines=o3d.utility.Vector2iVector([[0, 1]])
)

# Visualize the point cloud and the line
o3d.visualization.draw_geometries([pcd, keypoints, lines])

### PStttt — You can also fit these lines to the reflectance lines.