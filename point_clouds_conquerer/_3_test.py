from rich.console import Console
import open3d as o3d
import glob
import numpy as np
import matplotlib.pyplot as plt
from open3d.visualization.draw_plotly import get_plotly_fig


console = Console(force_terminal=True)
print = console.print 

# also we can use Matplotlib, Pypotree, Plotly
point_cloud_files = sorted(glob.glob("KITTI_PCD/*.pcd"))
idx = 357
print(point_cloud_files[idx])


point_cloud = o3d.io.read_point_cloud(point_cloud_files[idx])

points = np.asarray(point_cloud.points)
colors = np.asarray(point_cloud.colors)

print(points)
print(colors)

#########################################
#### Viewing using open3d
# o3d.visualization.draw_geometries([point_cloud])


#########################################
#### Viewing using pyqt
# from PyQt5.QtWidgets import QApplication
# import pyqtgraph.opengl as gl

# app = QApplication([])
# w = gl.GLViewWidget()
# plot = gl.GLScatterPlotItem(pos=points, color=colors, size=0.001, pxMode=False)
# w.setCameraPosition(distance=200, azimuth=45, elevation=30)
# w.addItem(plot)

# w.show()
# w.setWindowTitle('3D Point Cloud Visualization')
# app.exec_()


#########################################
##### Viewing using MeshLab


# import pymeshlab as ml
# import tempfile
# import subprocess
# import numpy as np



# # Create a MeshLab MeshSet
# ms = ml.MeshSet()

# # Convert numpy array to MeshLab-compatible mesh
# m = ml.Mesh(vertex_matrix=points)

# # Add the mesh to MeshLab for processing
# ms.add_mesh(m)

# # Save the processed mesh as a temporary PLY file
# with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_file:
#     temp_filename = temp_file.name  # Get the temp file path

# # Save the mesh for MeshLab visualization
# ms.save_current_mesh(temp_filename)

# # Path to MeshLab executable (Modify if needed)
# meshlab_path = "C:\\Program Files\\VCG\\MeshLab\\meshlab.exe"

# # Open the file in MeshLab
# subprocess.run([meshlab_path, temp_filename])

#########################################
#### Viewing using vedo
# import vedo


# # Create a vedo point cloud object
# vedo_cloud = vedo.Points(points, r=2, c=(0,1,0))

# # Display in vedo's interactive window
# vedo.show(vedo_cloud, "Point Cloud Visualization", axes=1)

#########################################
#### Viewing using plotly (not working)
# fig = get_plotly_fig(point_cloud, ...)
# fig.update_scenes(aspectmode='data')

# working
# import plotly.graph_objects as go

# distances = np.linalg.norm(points, axis=1)

# fig = go.Figure(data=[go.Scatter3d(
#     x=points[:, 0],
#     y=points[:, 1],
#     z=points[:, 2],
#     mode='markers',
#     marker=dict(
#         size=2,
#         color=distances,  # use distances for color
#         colorscale='Viridis',  # choose a colorscale Plasma, Viridis, Inferno, Cividis
#         colorbar=dict(title="Distance from Origin"),  # add a colorbar title
#         opacity=0.8
#     )
# )])
# fig.update_scenes(aspectmode='data')

# fig.show()




#########################################
#### adding camera
# import plotly.graph_objects as go

# distances = np.linalg.norm(points, axis=1)

# fig = go.Figure(data=[go.Scatter3d(
#     x=points[:, 0],
#     y=points[:, 1],
#     z=points[:, 2],
#     mode='markers',
#     marker=dict(
#         size=2,
#         color=distances,  # use distances for color
#         colorscale='Inferno',  # choose a colorscale
#         colorbar=dict(title="Distance from Origin", bgcolor="white"),  # add a colorbar title
#         opacity=0.8
#     )
# )])

# fig.update_layout(
#     scene=dict(
#         xaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
#         yaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
#         zaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
#         aspectmode='data',
#         camera=dict(
#             up=dict(x=-0.2, y=0, z=1),
#             center=dict(x=0.2, y=0, z=0.2),
#             eye=dict(x=-0.5, y=0, z=0.2))
#     ),
#     plot_bgcolor='black',
#     paper_bgcolor='black',
#     scene_dragmode='orbit'
# )
# fig.show()


#######################################
