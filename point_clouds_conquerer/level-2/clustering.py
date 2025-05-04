import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
# pcd = o3d.io.read_point_cloud("../data/PLAYGROUND/car.ply")
# pcd = o3d.io.read_point_cloud("../data/PLAYGROUND/bunny.pcd")
# pcd = o3d.io.read_point_cloud("../data/APOLLO/bg_1534313592.018882.pcd.ply")

pcd = o3d.io.read_point_cloud("../data/UDACITY/000000.pcd")
# pcd = o3d.io.read_point_cloud("../data/PLAYGROUND/turtlebot-pointcloud 2.ply")
# pcd = o3d.io.read_point_cloud("../data/KITTI_PCD/0000000200.pcd")


def ransac(pcd, distance_threshold=0.33, ransac_n=3, num_iterations=100):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0.5, 0.75, 0.25])
    inlier_cloud.paint_uniform_color([0.25, 0.5, 0.75])
    return inlier_cloud, outlier_cloud


def dbscan(outlier_cloud, eps=0.5, min_points=10):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(outlier_cloud.cluster_dbscan(
            eps=0.5, min_points=10, print_progress=False))
    # eps = 0.4 min_points=7
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(
        labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return outlier_cloud


def kdtree(outlier_cloud):
    # 5 BIS — How to find Objects using Clustering/ KDTREE
    pcd_tree = o3d.geometry.KDTreeFlann(outlier_cloud)

    # Initialize an array to keep track of assigned points
    assigned = np.zeros(len(outlier_cloud.points), dtype=bool)
    labels = np.zeros(len(outlier_cloud.points), dtype=int)
    print(len(labels))
    # colormap = plt.cm.get_cmap("viridis")  # Choose a colormap
    # Iterate over each point in the point cloud
    for i in range(len(outlier_cloud.points)):
        if assigned[i]:
            continue  # Skip points that are already assigned to a cluster

        query_point = outlier_cloud.points[i]  # Get the query point
        [k, indices, _] = pcd_tree.search_knn_vector_3d(
            query_point, 300)  # Perform KNN search

        # Process the nearest neighbors
        nearest_neighbors = np.asarray(outlier_cloud.points)[indices, :]
        # print(f"Query Point {i+1} - Nearest Neighbors:")
        # print(nearest_neighbors)

        # Assign the current point and its nearest neighbors to a cluster
        assigned[i] = True
        assigned[indices] = True
        # Assign a unique label to the cluster
        cluster_label = i + 1  # You can use any numbering scheme

        # Assign the label to the current point and its nearest neighbors
        labels[i] = cluster_label
        labels[indices] = cluster_label

    # Retrieve the unique cluster labels
    unique_labels = np.unique(labels)
    print(unique_labels)
    num_clusters = len(unique_labels)
    print(num_clusters)

    # Assign colors to each cluster based on their labels
    cluster_colors = plt.get_cmap("tab10")(np.linspace(0, 1, num_clusters))

    colors = np.zeros((len(outlier_cloud.points), 3))

    # Convert cluster labels to colors
    # Iterate over each cluster
    for cluster_label in unique_labels:
        # Get the indices of points in the cluster
        cluster_indices = np.where(labels == cluster_label)[0]
        print(cluster_indices)
        cluster_color = cluster_colors[cluster_label - 1]
        print(cluster_color)
        # Get the color for the cluster
        # Assign the color to all points in the cluster
        colors[cluster_indices] = cluster_color[:3]

    # colors = [cluster_colors[label - 1] for label in labels]  # Assign colors based on cluster labels

    # Set colors for the point cloud
    outlier_cloud.colors = o3d.utility.Vector3dVector(
        colors)  # cluster_colors[:, :3])

    # Visualize the colored point cloud
    return outlier_cloud


pcd = o3d.io.read_point_cloud("../data/UDACITY/000000.pcd")
inlier_cloud, outlier_cloud = ransac(
    pcd, distance_threshold=0.33, ransac_n=3, num_iterations=100)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
clustered_cloud = dbscan(outlier_cloud)
o3d.visualization.draw_geometries([inlier_cloud, clustered_cloud])
