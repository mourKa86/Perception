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
import torch.nn.functional as F
from torch.autograd import Variable

# custom imports
from data.lidar_od_scripts.gpuVersion.gpuVersion.visual_utils import plot_pc_data3d, plot_bboxes_3d
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PointNetDenseCls(nn.Module):
    """
    Network for Segmentation
    """
    def __init__(self, num_points = 2500, k = 2):
        super(PointNetDenseCls, self).__init__()
        self.num_points = num_points
        self.k = k
        self.feat = PointNetfeat(num_points, global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x, trans

def train_model(model, num_epochs, criterion, optimizer, dataloader_train,
                label_str='class_id', lr_scheduler=None, output_name='pointnet.pth'):
    # move model to device
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Starting {epoch + 1} epoch ...")

        # Training
        model.train()
        train_loss = 0.0
        for batch_dict in tqdm(dataloader_train, total=len(dataloader_train)):
            # Forward pass
            x = batch_dict['points'].transpose(1, 2).to(device)
            labels = batch_dict[label_str].to(device)
            pred, _ = model(x)
            loss = criterion(pred, labels)
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # adjust learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()

        # compute per batch losses, metric value
        train_loss = train_loss / len(dataloader_train)

        print(f'Epoch: {epoch + 1}, trainLoss:{train_loss:6.5f}')
    torch.save(model.state_dict(), output_name)

class PointNetCls(nn.Module):
    """
    Network for Classification: 512, 256, K.
    """
    def __init__(self, num_points = 2500, k = 2):
        super(PointNetCls, self).__init__()
        self.num_points = num_points
        self.feat = PointNetfeat(num_points, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1), trans

class PointNetfeat(nn.Module):
    """
    This is the T-Net for Feature Transform.
    There is also MLP part 64,128,1024.
    """
    def __init__(self, num_points = 2500, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans)
        x = x.transpose(2,1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans

class STN3d(nn.Module):
    """
    T-Net Model.
    STN stands for Spatial Transformer Network.
    """

    def __init__(self, num_points=2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.mp1 = torch.nn.MaxPool1d(num_points)

        # FC layers
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]

        # Expected input shape = (bs, 3, num_points)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

def collate_fn(batch_list):
    ret = {}
    ret['class_id'] =  torch.from_numpy(np.array([x['class_id'] for x in batch_list])).long()
    ret['class_name'] = np.array([x['class_name'] for x in batch_list])
    ret['points'] = torch.from_numpy(np.stack([x['points'] for x in batch_list], axis=0)).float()
    ret['seg_labels'] = torch.from_numpy(np.stack([x['seg_labels'] for x in batch_list], axis=0)).long()
    return ret

class ShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split_type, num_samples=2500):
        self.root_dir = root_dir
        self.split_type = split_type
        self.num_samples = num_samples
        with open(os.path.join(root_dir, f'{self.split_type}_split.json'), 'r') as f:
            self.split_data = json.load(f)

    def __getitem__(self, index):
        # read point cloud data
        class_id, class_name, point_cloud_path, seg_label_path = self.split_data[index]

        # point cloud data
        point_cloud_path = os.path.join(self.root_dir, point_cloud_path)
        pc_data = np.load(point_cloud_path)

        # segmentation labels
        # -1 is to change part values from [1-16] to [0-15]
        # which helps when running segmentation
        pc_seg_labels = np.loadtxt(os.path.join(self.root_dir, seg_label_path)).astype(np.int8) - 1
        #         pc_seg_labels = pc_seg_labels.reshape(pc_seg_labels.size,1)

        # Sample fixed number of points
        num_points = pc_data.shape[0]
        if num_points < self.num_samples:
            # Duplicate random points if the number of points is less than max_num_points
            additional_indices = np.random.choice(num_points, self.num_samples - num_points, replace=True)
            pc_data = np.concatenate((pc_data, pc_data[additional_indices]), axis=0)
            pc_seg_labels = np.concatenate((pc_seg_labels, pc_seg_labels[additional_indices]), axis=0)

        else:
            # Randomly sample max_num_points from the available points
            random_indices = np.random.choice(num_points, self.num_samples)
            pc_data = pc_data[random_indices]
            pc_seg_labels = pc_seg_labels[random_indices]

        # return variable
        data_dict = {}
        data_dict['class_id'] = class_id
        data_dict['class_name'] = class_name
        data_dict['points'] = pc_data
        data_dict['seg_labels'] = pc_seg_labels
        return data_dict

    def __len__(self):
        return len(self.split_data)

def main() -> None:
    DATA_FOLDER = 'data/Shapenetcore_benchmark/'

    class_name_id_map = {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4,
                         'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9,
                         'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13,
                         'Skateboard': 14, 'Table': 15}

    class_id_name_map = {v: k for k, v in class_name_id_map.items()}

    # ----- Shapenet Core Dataset exploration ------ #

    PCD_SCENE = dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode='data')
    NUM_PARTS = 16
    PART_COLORS = np.random.choice(range(255), size=(NUM_PARTS, 3)) / 255.0

    # ----- Split Data to Train, Val, Test ----- #
    train_set = ShapeNetDataset(root_dir = DATA_FOLDER, split_type='train')
    val_set = ShapeNetDataset(root_dir = DATA_FOLDER, split_type='val')
    test_set = ShapeNetDataset(root_dir = DATA_FOLDER, split_type='test')
    print(f"Train set length = {len(train_set)}")
    print(f"Validation set length = {len(val_set)}")
    print(f"Test set length = {len(test_set)}")

    # ----- Choosing a sample from the dataset ----- #
    data_dict = train_set[25]
    print(f"Keys in dataset sample = {list(data_dict.keys())}")
    points = data_dict['points']
    seg_labels = data_dict['seg_labels']
    print(f"class_id = {data_dict['class_id']}, class_name = {data_dict['class_name']}")

    # pc_plots = plot_pc_data3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], apply_color_gradient=False,
    #                           color=PART_COLORS[seg_labels - 1], marker_size=2)
    # layout = dict(template="plotly_dark",
    #               title=f"{data_dict['class_name']}, class id = {data_dict['class_id']}, from Shapenetcore Torch Dataset",
    #               scene=PCD_SCENE, title_x=0.5)
    # fig = go.Figure(data=pc_plots, layout=layout)
    # fig.show()

    # ----- From Numpy to Pytorch Tensor ----- #
    sample_loader = torch.utils.data.DataLoader(train_set, batch_size=16, num_workers=2, shuffle=True,
                                                collate_fn=collate_fn)
    dataloader_iter = iter(sample_loader)
    batch_dict = next(dataloader_iter)
    print(batch_dict.keys())
    for key in ['points', 'seg_labels', 'class_id']:
        print(f"batch_dict[{key}].shape = {batch_dict[key].shape}")

    # ----- All datasets from Numpy to Pytorch ----- #
    batchSize = 64
    workers = 2
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True, num_workers=workers,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batchSize, shuffle=True, num_workers=workers,
                                             collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchSize, shuffle=True, num_workers=workers,
                                              collate_fn=collate_fn)
    # ----- PointNet model ----- #
    # print('\nPointNet model')
    # test_model = STN3d().to(device)
    sim_data = Variable(torch.rand(32, 3, 2500)).to(device)
    # out = test_model(sim_data)
    # print('stn', out.size())
    #
    # pointfeat = PointNetfeat(global_feat=True).to(device)
    # out, _ = pointfeat(sim_data)
    # print('global feat', out.size())
    #
    # pointfeat = PointNetfeat(global_feat=False).to(device)
    # out, _ = pointfeat(sim_data)
    # print('point feat', out.size())
    #
    # cls = PointNetCls(k=16).to(device)
    # out, _ = cls(sim_data)
    # print('class', out.size())
    #
    # # ----- Training on PointNet ----- #
    # N_EPOCHS = 3
    # num_points = 2500
    # num_classes = 16
    # criterion = nn.NLLLoss()
    #
    # # create model, optimizer, lr_scheduler and pass to training function
    # num_classes = len(class_id_name_map.items())
    # classifier = PointNetCls(k=num_classes, num_points=num_points)
    #
    # # DEFINE OPTIMIZERS
    # optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    # if torch.cuda.is_available():
    #     classifier.cuda()
    #
    # _ = train_model(classifier, N_EPOCHS, criterion, optimizer, train_loader)
    #
    # # ----- Evaluation on PointNet (Inference) ----- #
    # classifier = PointNetCls(k=num_classes).to(device)
    # classifier.load_state_dict(torch.load('pointnet.pth'))
    # classifier.eval()
    #
    # total_loss = 0.0
    #
    # with torch.no_grad():
    #     for batch_dict in tqdm(test_loader, total=len(test_loader)):
    #         x = batch_dict['points'].transpose(1, 2).to(device)
    #         labels = batch_dict['class_id'].to(device)
    #         pred, _ = classifier(x)
    #
    #         # calculate loss
    #         loss = criterion(pred, labels)
    #         total_loss += loss.item()
    #
    # evaluation_loss = total_loss / len(test_loader)
    # print(evaluation_loss)
    #
    # # ----- Testing PointNet on Individual random Sample ----- #
    # # Random test sample
    # test_sample = test_set[np.random.choice(np.arange(len(test_set)))]
    # batch_dict = collate_fn([test_sample])
    # x = batch_dict['points'].transpose(1, 2).to(device)
    #
    # # Get model predictions
    # model_preds, _ = classifier(x)
    # predicted_class = torch.argmax(model_preds, axis=1).detach().cpu().numpy()[0]
    # predicted_class_name = class_id_name_map[predicted_class]
    # pred_class_probs = F.softmax(model_preds.flatten(), dim=0).detach().cpu().numpy()
    #
    # # plot results
    # title = f"Label = {test_sample['class_name']}, Predicted class = {predicted_class_name}"
    # fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter3d"}, {}]], column_widths=[0.4, 0.6])
    # fig.update_layout(template="plotly_dark", scene=PCD_SCENE, height=400, width=1200,
    #                   title=title, title_x=0.1, title_y=0.97, margin=dict(r=0, b=0, l=0, t=0))
    # fig.add_trace(
    #     plot_pc_data3d(x=test_sample['points'][:, 0], y=test_sample['points'][:, 1], z=test_sample['points'][:, 2]),
    #     row=1, col=1)
    # fig.add_trace(go.Bar(x=list(class_name_id_map.keys()), y=pred_class_probs, showlegend=False), row=1, col=2)
    # fig.show()

    # # ----- Segmentation ----- #
    seg = PointNetDenseCls(k=16).to(device)
    print(seg)
    out, _ = seg(sim_data)
    print('seg', out.size())

    # ----- Training on Segmentation ----- #
    N_EPOCHS = 3
    num_points = 2500
    criterion = nn.CrossEntropyLoss()

    # create model, optimizer, lr_scheduler and pass to training function
    num_classes = len(class_id_name_map.items())
    dense_classifier = PointNetDenseCls(k=NUM_PARTS, num_points=num_points)
    dense_classifier.to(device)

    # DEFINE OPTIMIZERS
    optimizer = optim.SGD(dense_classifier.parameters(), lr=0.01, momentum=0.9)

    train_model(dense_classifier, N_EPOCHS, criterion, optimizer, train_loader,
                label_str='seg_labels', output_name='pointnet_seg.pth')

    # ----- Evaluation on Segmentation (Inference) ----- #
    dense_classifier.load_state_dict(torch.load('pointnet_seg.pth'))
    dense_classifier.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch_dict in tqdm(test_loader, total=len(test_loader)):
            x = batch_dict['points'].transpose(1, 2).to(device)
            labels = batch_dict['seg_labels'].to(device)
            pred, _ = dense_classifier(x)

            # calculate loss
            loss = criterion(pred, labels)
            total_loss += loss.item()

    evaluation_loss = total_loss / len(test_loader)
    print(evaluation_loss)

    # ----- Test on individual items ----- #
    # Random test sample
    test_sample = test_set[np.random.choice(np.arange(len(test_set)))]
    batch_dict = collate_fn([test_sample])

    # Get model predictions
    x = batch_dict['points'].transpose(1, 2).to(device)
    model_preds, _ = dense_classifier(x)
    pred_part_labels = torch.argmax(model_preds, axis=1).detach().cpu().numpy()[0]

    points = test_sample['points']
    part_labels = test_sample['seg_labels']

    # plot results
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
                        column_widths=[0.5, 0.5],
                        subplot_titles=('Part Labels', 'Part Predictions'))

    # ground truth part labels
    part_label_plots = plot_pc_data3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], apply_color_gradient=False,
                                      color=PART_COLORS[part_labels - 1], marker_size=2)

    # ground truth part labels
    pred_part_label_plots = plot_pc_data3d(x=points[:, 0], y=points[:, 1], z=points[:, 2], apply_color_gradient=False,
                                           color=PART_COLORS[pred_part_labels - 1], marker_size=2)

    fig.update_layout(template="plotly_dark", scene=PCD_SCENE, scene2=PCD_SCENE, height=400, width=1200,
                      title='PointNet Segmentation', title_x=0.5, title_y=0.97, margin=dict(r=0, b=0, l=0, t=0))
    fig.add_trace(part_label_plots, row=1, col=1)
    fig.add_trace(pred_part_label_plots, row=1, col=2)
    fig.show()

if __name__ == '__main__':
    main()

