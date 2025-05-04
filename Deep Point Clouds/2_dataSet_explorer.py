# Usual Imports
import os
import sys
import json
import numpy as np
from tqdm import tqdm
import glob
import random

# plotting library
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# append path to custom scripts
sys.path.append('data/lidar-od-scripts/gpuVersion/gpuVersion/')

# DL Imports
import torch
import torch.nn as nn

# custom imports
from data.lidar_od_scripts.gpuVersion.gpuVersion.visual_utils import plot_pc_data3d, plot_bboxes_3d

def main() -> None:
    DATA_FOLDER = 'data/Shapenetcore_benchmark/'

    class_name_id_map = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4,
                         'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9,
                         'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13,
                         'Skateboard': 14, 'Table': 15}

    class_id_name_map = {v: k for k, v in class_name_id_map.items()}

    # ----- Shapenet Core Dataset exploration ------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PCD_SCENE = dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data')

    # ----- Train Set Exploration ----- #
    train_split_data = json.load(open('data/Shapenetcore_benchmark/train_split.json', 'r'))
    train_class_count = np.array([x[0] for x in train_split_data])

    # ----- Plot Train Distribution ----- #
    train_dist_plots = [go.Bar(x=list(class_name_id_map.keys()), y=np.bincount(train_class_count))]
    layout = dict(template="plotly_dark", title="Shapenet Core Train Distribution", title_x=0.5)
    fig = go.Figure(data=train_dist_plots, layout=layout)
    fig.show()

    # ----- list all point clouds ----- #
    points_list = glob.glob("data/Shapenetcore_benchmark/04379243/points/*.npy")
    print(len(points_list))

    # ----- select random point cloud ----- #
    idx = random.randint(0, len(points_list))

    # ----- load point cloud ----- #
    points = np.load(points_list[idx])
    print(f"points shape = {points.shape}, min xyz = {np.min(points, axis=0)}, max xyz = {np.max(points, axis=0)}")

    # ----- load point cloud and segmentation labels ----- #
    seg_file_path = points_list[idx].replace('points', 'points_label').replace('.npy', '.seg')
    seg_labels = np.loadtxt(seg_file_path).astype(np.int8)
    print(f"seg_labels shape = {seg_labels.shape}, unique labels = {np.unique(seg_labels)}")

    # ----- plot point cloud ----- #
    # there are max of 16 parts in an object in Shapenet core dataset
    # creating random colors in according to part label
    NUM_PARTS = 16
    PART_COLORS = np.random.choice(range(255), size=(NUM_PARTS, 3)) / 255.0
    pc_plots = plot_pc_data3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], apply_color_gradient=False,
                              color=PART_COLORS[seg_labels - 1], marker_size=2)
    layout = dict(template="plotly_dark", title="Raw Point cloud", scene=PCD_SCENE, title_x=0.5)
    fig = go.Figure(data=pc_plots, layout=layout)
    fig.show()


if __name__ == '__main__':
    main()

