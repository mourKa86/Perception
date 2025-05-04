import open3d as o3d
import glob
from rich import print
import numpy as np

def down_sample(pcd, voxel_size):    
    print(f"Points before downsampling: {len(pcd.points)} ")

    pcd = pcd.voxel_down_sample(voxel_size=0.1)

    print(f"Points after downsampling: {len(pcd.points)}")
    # o3d.visualization.draw_geometries([pcd])
    return pcd

def extract_voxel_centers(voxel_grid):
    """Extracts voxel centers from the voxel grid"""
    voxels = voxel_grid.get_voxels()
    centers = [voxel.grid_index for voxel in voxels]  # Extract voxel indices
    return np.array(centers, dtype=np.float32)

def view_pcd(points):
    from PyQt5.QtWidgets import QApplication
    import pyqtgraph.opengl as gl

    print("Type of points:", type(points))
    print("Shape of points:", points.shape)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Error: `points` must have shape (N, 3)")

    points = np.asarray(points, dtype=np.float32)

    app = QApplication([])
    w = gl.GLViewWidget()
    plot = gl.GLScatterPlotItem(pos=points, size=3, pxMode=True)
    w.setCameraPosition(distance=200, azimuth=45, elevation=30)
    w.addItem(plot)

    w.show()
    w.setWindowTitle('3D Point Cloud Visualization')
    app.exec_()

## voxeuzmmn I
if __name__ == "__main__":
    # pcd = o3d.io.read_point_cloud("data\\PLAYGROUND\\turtlebot-pointcloud 2.ply")
    point_cloud_files = sorted(glob.glob("KITTI_PCD/*.pcd"))
    pcd = o3d.io.read_point_cloud(point_cloud_files[200])
    
    voxel_size = 0.01
    voxel_down_sample = down_sample(pcd, voxel_size)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(voxel_down_sample, voxel_size=voxel_size)
    # o3d.visualization.draw_geometries([voxel_grid])

    points = extract_voxel_centers(voxel_grid)
    # colors = np.asarray(voxel_down_sample.colors)



    view_pcd(points)

