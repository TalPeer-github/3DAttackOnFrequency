import os
import random
import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# from pyvis.network import Network
# from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import knn_graph, radius_graph


def create_dataset(dataset_name, pre_transform, transform, name='40', num_point_to_sample=2048):
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
    pre_transform = get_pre_transform()
    transform = get_transform()
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


def get_transform(num_sample=256):
    """
    It's the transform that is applied to the data object before every access.
    In this case, we use the torch_geometric.transforms.SamplePoints to sample a fixed number of points
    on the mesh faces according to their face area.
    :param dataset_name: modelnet dataset to use - either 40/10. default is set to 40.
    :return:
    """
    transform = T.Compose([T.SamplePoints(num_sample), T.KNNGraph(k=6)])
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


def visualize_points(pc_coords):
    points = torch.tensor(pc_coords).numpy()
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o', s=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Point Cloud')

    plt.show()


def get_seq_random_walk_local_jumps(mesh_extra, f0, seq_len):
    """
    for an input set S_i we generate the walk W_ij of length l as follows:
    1. the walk origin point, p0, is randmoly selected from the set S_i.
    2. point are iteratively added to the walk by selecting random point from the set of k-nearest neighbors of the last
        point in the sequence (exluding neighbors who were added at an earlier stage)
    3.  In the rare case where all k nearest neighbors were already added to the walk, a new
        random un-visited point is chosen and the walk generation proceeds as before.
    Choosing the closest neighbor imposes a strong constraint on the generation process
    and reduces the randomness and the ability to visit sparser regions.

     KDtree is an efficient, hierarchical space partitioning data structure for nearest neighbors queries,
     in which every node represents an axis-aligned hyper-rectangle and contains the set of points in it.
    :param mesh_extra:
    :param f0:
    :param seq_len:
    :return:
    """
    kdtree = KDTree()
    n_vertices = mesh_extra['kdtree_query'].shape[0]
    kdtr = mesh_extra['kdtree_query']

    seq = np.zeros((seq_len + 1,), dtype=np.int32) - 1
    jumps = np.zeros((seq_len + 1,), dtype=np.bool)
    seq[0] = f0
    visited = np.zeros((n_vertices + 1,), dtype=np.bool)
    visited[-1] = True
    visited[f0] = True
    for i in range(1, seq_len + 1):
        to_consider = [n for n in kdtr[seq[i - 1]] if not visited[n]]
        if len(to_consider):
            random_point = np.random.choice(to_consider)
            seq[i] = random_point
            jumps[i] = False
        else:
            seq[i] = np.random.randint(n_vertices)
            jumps[i] = True
            visited = np.zeros((n_vertices + 1,), dtype=np.bool)
            visited[-1] = True
        visited[seq[i]] = True

    return seq, jumps


_, val_dataset = create_dataset(dataset_name="ModelNet10", name='10', pre_transform=None, transform=None)
for pc in val_dataset[::-1]:
    print(pc.pos)
    visualize_points(pc.pos)
    break
