import os
import copy 
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints, NormalizeScale, BaseTransform
from torch_geometric.data import DataLoader, InMemoryDataset
from torch_geometric.nn import EdgeConv, knn_graph
import torch_geometric.transforms as T

from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform

from torch_geometric.nn import global_max_pool
from torch_geometric.utils import to_networkx
import torch.nn as nn

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

default_args = {
    "k": 10,
    "n_sample": 2048,
    "num_walks": 32,  
    "walks_len": 128,
    "batch_size": 16,
    "num_workers": 4  
}


class MeshCNNClassifier(torch.nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, num_classes=10):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels * 2), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, data):
        x = data.pos  # placeholder for edge-based features if added
        x = self.conv1(x)
        x = self.conv2(x)
        x = global_max_pool(x, data.batch)
        return self.fc(x)

def get_datasets(root='data/ModelNet10'):
    transform = T.Compose([NormalizeScale(), SamplePoints(2048, remove_faces=False)])

    train_dataset = ModelNet(root=root, name='10', train=True, transform=transform)
    test_dataset = ModelNet(root=root, name='10', train=False, transform=transform)

    return train_dataset, test_dataset

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

def run_training(save_model=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = MeshCNNClassifier(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 21):
        loss = train(model, train_loader, optimizer, device)
        acc = test(model, test_loader, device)
        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

    save_model = True
    last_model_state_dict = model.state_dict()
    model_path = "victim_models/checkpoints/"
    model_id = "MeshCNN_84_20epochs_16bs_1e3lr" # <ModelName>_<DayMonth>_<HourMinutes>
    if save_model:
        last_saved_model = f'{model_path}{model_id}.pth'
        torch.save(model.state_dict(),last_saved_model)