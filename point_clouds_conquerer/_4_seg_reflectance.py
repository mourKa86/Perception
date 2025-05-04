from rich.console import Console
import open3d as o3d
import glob
import numpy as np
import matplotlib.pyplot as plt
from open3d.visualization.draw_plotly import get_plotly_fig

import cv2
from tqdm import tqdm


console = Console(force_terminal=True)
print = console.print 

# also we can use Matplotlib, Pypotree, Plotly

def vis_pcd(point_cloud, save="False", show=True, ):
    fig = get_plotly_fig(point_cloud, width = 800, height = 533, mesh_show_wireframe =False,
                            point_sample_factor = 1, front = (1,1,1), lookat =(1,1,1), up=(1,1,1), zoom=1.0)
    #fig.update_scenes(aspectmode='data')
    fig.update_layout(
    scene=dict(
        xaxis=dict(visible=False,range=[-70,70]),
        yaxis=dict(visible=False,range=[-40,40]),
        zaxis=dict(visible=False,range = [-5,1]),
        aspectmode='manual', aspectratio= dict(x=2, y=1, z=0.1),
        #aspectmode="data",
        camera=dict(
            up = dict(x=0.15, y =0, z=1),
            center=dict(x=0, y=0, z=0.1),
            eye = dict(x = -0.3, y=0, z=0.2)
            # up = dict(x=0.05, y=0, z=1),
            # center = dict(x=0.2, y=0,z=0.05),
            # eye=dict(x= -0.15, y=0, z=0.05),
        )
    ),
    plot_bgcolor='black',
    paper_bgcolor='black',
    scene_dragmode='orbit'
)
    if show == True:
        fig.show()

    if save != "False":
        fig.write_image("output/"+save+"_processed.jpg", scale=3)

    return fig


def create_video(point_cloud_files):
    output_handle = cv2.VideoWriter("reflectance_output.avi", cv2.VideoWriter_fourcc(* 'DIVX'), 10, (2400, 1599))

    start_index = 350
    stop_index = 400

    pbar = tqdm(total = (stop_index - start_index), position=0, leave=True)
    all_pcd = [o3d.io.read_point_cloud(point_cloud_files[i]) for i in range(start_index, stop_index)]
    
    for i in range(len(all_pcd)):
        roi_pcd = lane_line_pipeline(all_pcd[i])
        fig = vis_pcd([all_pcd[i].paint_uniform_color((0.2,0.4,0.2)), roi_pcd], show=False, save=str(start_index+i))
        output_handle.write(cv2.imread("output/"+str(start_index+i)+"_processed.jpg"))
        pbar.update(1)

    output_handle.release()

def lane_line_pipeline(pcd):
    filtered_point_cloud = reflectivity_threshold(pcd)
    roi_pcd = roi_filter(filtered_point_cloud)

    return roi_pcd

def reflectivity_threshold(pcd, thresh=0.45):
    colors = np.asarray(pcd.colors)
    reflectivities = colors[:, 0]
    # Get the point coordinates
    points = np.asarray(pcd.points)
    # Create a mask of points that have reflectivity above the threshold
    mask = reflectivities > thresh

    # Filter points and reflectivities using the mask
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    # Create a new point cloud with the filtered points
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_point_cloud

def roi_filter(pcd, roi_min=(-20,-3,-2), roi_max=(20,3,0)):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    mask_roi = np.logical_and.reduce((
        points[:, 0] >= roi_min[0],
        points[:, 0] <= roi_max[0],
        points[:, 1] >= roi_min[1],
        points[:, 1] <= roi_max[1],
        points[:, 2] >= roi_min[2],
        points[:, 2] <= roi_max[2]
    ))

    roi_points = points[mask_roi]
    roi_colors = colors[mask_roi]

    # Create a new point cloud with the filtered points
    roi_pcd = o3d.geometry.PointCloud()

    roi_pcd.points = o3d.utility.Vector3dVector(roi_points)
    roi_pcd.colors = o3d.utility.Vector3dVector(roi_colors)

    return roi_pcd

# if __name__ == "__main__":
    point_cloud_files = sorted(glob.glob("KITTI_PCD/*.pcd"))
    idx = 357
    roi_pcd, filtered_point_cloud, pcd = lane_line_pipeline(point_cloud_files[idx])
    fig = vis_pcd([pcd.paint_uniform_color((0.2,0.2,0.2)), roi_pcd], show=True)

    # create_video(point_cloud_files)