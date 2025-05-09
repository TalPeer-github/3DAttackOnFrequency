import os
import json
import random
import time
import glob
import tqdm 

import numpy as np 
import pandas as pd
import seaborn as sns
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D

import torch_geometric

from torch_geometric.data import Dataset
from torch_geometric.datasets import StochasticBlockModelDataset
import torch_geometric.transforms as T
from torch_geometric.transforms import BaseTransform, KNNGraph, RandomNodeSplit
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, fps
from torch_geometric.nn.pool import global_max_pool
from torch_geometric.utils import to_networkx, to_dense_adj, dropout

import torch_cluster
import torch

from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.distributions import Categorical

from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

torch.manual_seed(seed=42)
from utils.config import default_args, filtered_classes,manual_probs,modelnet10_classes,modelnet10_labels

def plot_class_wise_scores(class_names, scores, score_name, eps):
    fig, ax = plt.subplots(1,1)
    x_axis = np.arange(scores.shape[0])
    for i in range(scores.shape[0]):
        ax.bar([x_axis[i]], [scores[i]], label=class_names[i])
    ax.set_xlabel("class number")
    ax.set_ylabel(score_name)
    plt.legend(loc=(1.04,0))
    #plt.savefig(score_name + str(eps) + '.png')
    plt.show()
    

def compute_centrality_measures(data):
    G = torch_geometric.utils.to_networkx(data, to_undirected=True)

    centralities = {
        'degree': nx.degree_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
    }

    return G, centralities


def visualize_degree_hist(pc_data,hist_stat='density',element='bars'):
    walks_idx = pc_data.walks_idx
    label = modelnet10_classes[pc_data.y.item()].capitalize()
    
    walks_idxs = torch.cat(torch.tensor(walks_idx, dtype=torch.long).unbind(0))
    degrees = torch_geometric.utils.degree(walks_idxs, dtype=torch.long).numpy()

    linewidth = 1.5 if element == 'bars' else 2.5
    sns.histplot(degrees,bins=12,stat=hist_stat, element=element,facecolor='#CDE8E5',edgecolor='#006A71',linewidth=linewidth)
    
    plt.tight_layout()
    plt.title(f"{label}\n\nRandom Walks degree distribution")
    plt.show()

def visualize_value_counts_degree_distribution(pc_data,stat='probability'):
    pc_class = modelnet10_classes[pc_data.y.item()].capitalize()
    edge_idx = torch.cat(pc_data.edge_index.unbind(0))
    edge_df = pd.DataFrame(edge_idx.numpy())
    edge_vc = edge_df.value_counts().reset_index().rename(columns={0:"node_idx"})
    edge_vc_scaleinv = edge_vc[edge_vc['count']>=10]
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    sns.histplot(edge_vc['count'], stat=stat, bins=20, element='bars',facecolor='#CDE8E5', edgecolor='#006A71', linewidth=1.5, ax=axes[0])
    axes[0].set_title("Degree occurrences (full)")
    axes[0].set_xticks(range(int(edge_vc['count'].min()), int(edge_vc['count'].max()) + 1))
    sns.histplot(edge_vc_scaleinv['count'], stat=stat, bins=20, element='bars',facecolor='#CDE8E5', edgecolor='#006A71', linewidth=1.5, ax=axes[1])
    axes[1].set_title("Degree occurrences (degree > 10)")
    axes[1].set_xticks(range(int(edge_vc_scaleinv['count'].min()), int(edge_vc_scaleinv['count'].max()) + 1))

    plt.suptitle(f"{pc_class}\nRandom Walks Generated Graph (Degree ~ Power law)")
    plt.show()
    

def visualize_communities(pc_data):
    pos = pc_data.pos.cpu().numpy()
    edge_index = pc_data.edge_index.cpu().numpy()
    edge_attr = pc_data.edge_attr.cpu().numpy()
    pc_class = modelnet10_classes[pc_data.y.item()].capitalize()
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    walks_batches = [range(0,11)] * 3
    colors = matplotlib.colormaps["Paired"] 
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': '3d'})
    
    for idx, batch in enumerate(walks_batches):
        ax = axes[idx]
        for i in batch:
            color = colors(i)
            walk_idx = min(idx * 11 + i,31)
            edge_mask = edge_attr[:, walk_idx] == True
            selected_edges = edge_index[:, edge_mask].T

            us,vs = selected_edges[:,0],selected_edges[:,1]
            u_xs, u_ys, u_zs = pos[us, 0], pos[us, 1], pos[us, 2]
            v_xs, v_ys, v_zs = pos[vs, 0], pos[vs, 1], pos[vs, 2]
            ax.scatter([u_xs, v_xs], [u_ys, v_ys], [u_zs, v_zs], s=20, c='gray',alpha=0.2, )
            for i, j in selected_edges:
                 ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], c=color, alpha=1)


        ax.set_title(f'{pc_class.capitalize()}')

    
    plt.tight_layout()
    plt.show()


def visualize_3d_from_pos(pc_data, plot_by_centrality='closeness'):
    pos = pc_data.pos.cpu().numpy()
    pc_class = modelnet10_classes[pc_data.y.item()].capitalize()
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    G, centrality = compute_centrality_measures(pc_data)
    node_alpha = [centrality[plot_by_centrality][node] for node in G.nodes()]

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(xs, ys, zs, s=50, c=node_alpha, cmap='seismic')

    edge_index = pc_data.edge_index.cpu().numpy()
    for i, j in edge_index.T:
        # edge_alpha = 0.5 * (node_alpha[i] + node_alpha[j])
        edge_alpha = min(node_alpha[i], node_alpha[j])
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], c='gray', alpha=edge_alpha)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f'{plot_by_centrality.capitalize()}')

    ax.set_title(f'{pc_class} - 3D Point Cloud')
    plt.tight_layout()
    plt.show()

def viz_fps(pos):
    
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, s=100,)
    ax.set_title(f'{"FPS"} sampled points 3D Point Cloud')
    plt.tight_layout()
    plt.show()

def visualize_3d_centralities(pc_data):
    pos = pc_data.pos.cpu().numpy()
    pc_class = modelnet10_classes[pc_data.y.item()].capitalize()
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]

    G, centrality = compute_centrality_measures(pc_data)
    centralities = ['degree', 'closeness', 'betweenness']
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': '3d'})
    
    for idx, centrality_type in enumerate(centralities):
        ax = axes[idx]
        node_alpha = [centrality[centrality_type][node] for node in G.nodes()]
        scatter = ax.scatter(xs, ys, zs, s=40, c=node_alpha, cmap='RdYlBu', alpha=1.)
        edge_index = pc_data.edge_index.cpu().numpy()
        for i, j in edge_index.T:
            edge_alpha = max(node_alpha[i],node_alpha[j])
            edge_alpha = (edge_alpha - np.min(node_alpha)) / (np.max(node_alpha) - np.min(node_alpha))
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], [zs[i], zs[j]], c='gray', alpha=edge_alpha)
        
        ax.set_title(f'{pc_class.capitalize()} - {centrality_type.capitalize()}')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(f'{centrality_type.capitalize()}')
    
    plt.tight_layout()
    plt.show()