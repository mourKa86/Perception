import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import open3d as o3d
import glob
import numpy as np
from open3d.visualization.draw_plotly import get_plotly_fig
import webbrowser

# Function to visualize point cloud
def vis_pcd(point_cloud, show=True):
    fig = get_plotly_fig(point_cloud, width=800, height=533, mesh_show_wireframe=False,
                         point_sample_factor=1, front=(1,1,1), lookat=(1,1,1), up=(1,1,1), zoom=1.0)
    
    # Set scene layout and camera parameters
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-70, 70]),
            yaxis=dict(visible=False, range=[-40, 40]),
            zaxis=dict(visible=False, range=[-5, 1]),
            aspectmode='manual',
            aspectratio=dict(x=2, y=1, z=0.1),
            camera=dict(
                # up=dict(x=0.15, y=0, z=1),
                # center=dict(x=0, y=0, z=0.1),
                # eye=dict(x=-0.3, y=0, z=0.2),
                up=dict(x=0.05, y=0, z=1),
                center=dict(x=0.2, y=0, z=0.05),
                eye=dict(x=-0.15, y=0, z=0.05)
            )
        ),
        plot_bgcolor='black',
        paper_bgcolor='black',
        scene_dragmode='orbit'
    )
    
    if show:
        fig.show()

    return fig

# Load point cloud file
point_cloud_files = sorted(glob.glob("KITTI_PCD/*.pcd"))
idx = 357
point_cloud = o3d.io.read_point_cloud(point_cloud_files[idx])

# Generate visualization
fig = vis_pcd([point_cloud])

# Initialize Dash app
app = dash.Dash()
app.layout = html.Div([
    html.Div(id="output"),
    dcc.Graph(id="fig", figure=fig)  # Display the 3D plot
])

# Callback to track camera movements
@app.callback(
    Output("output", "children"),
    [Input("fig", "relayoutData")]
)
def update_output(data):
    if data:
        print("Camera Position Update:", data)  # Print to VS Code terminal
    return str(data)  # Display in Dash UI

# Open browser automatically
if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:8050/")
    app.run_server(debug=True, use_reloader=False)
