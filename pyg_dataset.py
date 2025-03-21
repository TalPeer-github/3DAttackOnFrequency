import os
import random
import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import knn_graph, radius_graph


def create_dataset(dataset_name, pre_transform, transform, name='40',num_point_to_sample=2048):
    """
    The input pipeline PC classification task is not much different from a standard PyTorch-based input pipeline.
    However, we would be using PyTorch Geometrics dataset API for loading the ModelNet datasets which makes loading
    and processing these datasets way easier for us.
    :return:
    """
    dataset_name = f"ModelNet{name}" if dataset_name is None else dataset_name

    if pre_transform is None:
        pre_transform = T.NormalizeScale()
    if transform is None:
        transform = T.SamplePoints(num_point_to_sample)

    train_dataset = ModelNet(
        root=dataset_name, name=name, train=True,
        transform=transform, pre_transform=pre_transform
    )
    val_dataset = ModelNet(
        root=dataset_name, name=name, train=False,
        transform=transform, pre_transform=pre_transform
    )
    return train_dataset, val_dataset


def get_pre_transform():
    """
    Pre Transform: It's the transform applied to the data object before being saved to the disk.
     this case, we apply the torch_geometric.transforms.NormalizeScale to the data as a pre-transform
     that centers and normalizes node positions in the 3D mesh to the interval (−1,1)(-1, 1)(−1,1).
    :return:
    """
    pre_transform = T.NormalizeScale()
    return pre_transform


def get_transform(num_sample=2048):
    """
    It's the transform that is applied to the data object before every access.
    In this case, we use the torch_geometric.transforms.SamplePoints to sample a fixed number of points
    on the mesh faces according to their face area.
    :param dataset_name: modelnet dataset to use - either 40/10. default is set to 40.
    :return:
    """
    transform = T.SamplePoints(num_sample)
    return transform


def create_dataloaders(train_dataset, val_dataset):
    """
    Create dataloaders from the datasets which merges the data objects into mini-batches and wrap an iterable around it.
    We use the torch_geometric.loader.DataLoader to create our dataloaders
    :return:
    """
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers
    )
    return train_loader, val_loader
