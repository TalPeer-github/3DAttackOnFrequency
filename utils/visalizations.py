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

import scipy
from scipy.spatial import KDTree, Delaunay

import matplotlib
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

torch.manual_seed(seed=0)
from config import args, modelnet10_labels, modelnet10_classes, label_to_class, class_to_label, default_args, filtered_classes,manual_probs

def plot_pc(pos_orig, pos_adv, true_class, pred_class, adv_class):
    pos_orig = pos_orig.detach().cpu().numpy()
    pos_adv = pos_adv.detach().cpu().numpy()
    true_class = class_to_label[true_class.detach().cpu().numpy().item()].capitalize()
    pred_class = class_to_label[pred_class.detach().cpu().numpy().item()].capitalize()
    adv_class = class_to_label[adv_class.detach().cpu().numpy().item()].capitalize()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), subplot_kw={'projection': '3d'}, )
    axes[0].scatter(pos_orig[:, 0], pos_orig[:, 1], pos_orig[:, 2], edgecolor='#670D2F', s=50, alpha=0.5, c='#EF88AD',
                    linewidth=1.25)
    axes[0].set_title(f"Clean Prediction: {pred_class}")

    axes[1].scatter(pos_adv[:, 0], pos_adv[:, 1], pos_adv[:, 2], edgecolor='#670D2F', s=50, alpha=0.5, c='#EF88AD',
                    linewidth=1.25)
    axes[1].set_title(f"Attack Prediction: {adv_class}")

    plt.suptitle(f"Adversarial Point Cloud (True label = {true_class})", fontsize=16, fontweight='bold')

    exp_name = args.experiment_name
    fig_path = f"attacks/visualizations/{exp_name}/{true_class}/"
    os.makedirs(fig_path, exist_ok=True)

    img_title = f"{fig_path}_clean_{pred_class}_adv_{adv_class}.png"
    plt.savefig(img_title)


def visualize_point_cloud_spectral(pos_orig, pos_adv, true_class="", pred_class="", adv_class="", title_orig="",
                                   title_adversarial="", spectral=True, linf=True):
    def plot_mesh(pos, adv=False):
        points = pos
        tri = Delaunay(points)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        verts = pos[tri.simplices]

        mesh = Poly3DCollection(verts, alpha=0.4, edgecolor='#670D2F', facecolor='#EF88AD')
        ax.add_collection3d(mesh)

        ax.set_xlim(pos[:, 0].min(), pos[:, 0].max())
        ax.set_ylim(pos[:, 1].min(), pos[:, 1].max())
        ax.set_zlim(pos[:, 2].min(), pos[:, 2].max())

        exp_name = args.experiment_name
        fig_path = f"attacks/visualizations/{exp_name}/{true_class}/"
        os.makedirs(fig_path, exist_ok=True)
        mesh_type = "mesh_adv" if adv else "mesh_original"
        img_title = f"{fig_path}{mesh_type}_clean_{pred_class}_adv_{adv_class}.png"
        plt.savefig(img_title)

    true_class = true_class.split(' ')[0]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12),
                             subplot_kw={'projection': '3d'}, )  # height_ratios=[0.6,0.4])

    axes[0, 0].scatter(pos_orig[:, 0], pos_orig[:, 1], pos_orig[:, 2], edgecolor='#670D2F', s=50, alpha=0.5,
                       c='#EF88AD', linewidth=1.25)
    axes[0, 0].set_title(f"Clean Prediction: {pred_class}")

    axes[0, 1].scatter(pos_adv[:, 0], pos_adv[:, 1], pos_adv[:, 2], edgecolor='#670D2F', s=50, alpha=0.5, c='#EF88AD',
                       linewidth=1.25)
    axes[0, 1].set_title(f"Attack Prediction: {adv_class}")

    plt.suptitle(f"Attacks Point Cloud & Mesh: {true_class.capitalize()}", fontsize=16, fontweight='bold')

    tri_orig = Delaunay(pos_orig)
    tri_adv = Delaunay(pos_adv)

    verts_orig = pos_orig[tri_orig.simplices]
    verts_adv = pos_adv[tri_adv.simplices]

    edgecolor = '#670D2F'
    mesh_orig = Poly3DCollection(verts_orig, alpha=0.25, edgecolor='#670D2F', facecolor='#EF88AD')
    mesh_adv = Poly3DCollection(verts_adv, alpha=0.25, edgecolor='#670D2F', facecolor='#EF88AD')

    axes[1, 0].add_collection3d(mesh_orig)
    axes[1, 0].set_xlim(pos_orig[:, 0].min(), pos_orig[:, 0].max())
    axes[1, 0].set_ylim(pos_orig[:, 1].min(), pos_orig[:, 1].max())
    axes[1, 0].set_zlim(pos_orig[:, 2].min(), pos_orig[:, 2].max())

    axes[1, 1].add_collection3d(mesh_adv)
    axes[1, 1].set_xlim(pos_adv[:, 0].min(), pos_adv[:, 0].max())
    axes[1, 1].set_ylim(pos_adv[:, 1].min(), pos_adv[:, 1].max())
    axes[1, 1].set_zlim(pos_adv[:, 2].min(), pos_adv[:, 2].max())

    attack_type = "linf" if linf else "l2"  # spectral" if spectral else "noise"
    exp_name = args.experiment_name
    fig_path = f"attacks/visualizations/{exp_name}/{true_class}"
    os.makedirs(fig_path, exist_ok=True)
    img_title = f"{fig_path}/{attack_type}_merged_clean_{pred_class}_adv_{adv_class}.png"
    plt.savefig(img_title)
    plt.close()


def plot_attack(orig_pos, adv_pos, clean_pred, adv_pred, true_label, linf=False):
    true_class = f"{modelnet10_classes[true_label]}"
    pred_class = f"{modelnet10_classes[clean_pred]}"
    adv_class = f"{modelnet10_classes[adv_pred]}"

    title_orig = f"Original Point Cloud (Class: {true_class}) \n\n Model Prediction: {pred_class}"
    title_adversarial_pred = f"Adversarial Prediction (Class: {adv_class})"

    visualize_point_cloud_spectral(pos_orig=orig_pos, pos_adv=adv_pos, true_class=true_class, pred_class=pred_class,
                                   adv_class=adv_class,
                                   title_orig=title_orig, title_adversarial=title_adversarial_pred, spectral=True,
                                   linf=linf)


def plot_mesh_from_arrays(vertices, faces, show=False, save_path=None):
    faces = faces.astype(int)  # ensure indices are integers
    corners = vertices[faces]  # shape: (F, 3, 3)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(corners, alpha=0.25, edgecolor='#670D2F', facecolor='#EF88AD')
    ax.add_collection3d(mesh)

    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               edgecolor='#670D2F', s=50, alpha=0.3, c='#EF88AD', linewidth=1.25)

    max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max() / 2
    mid = vertices.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_box_aspect([1, 1, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}.png", bbox_inches='tight')

    if show:
        plt.show()
    plt.close()


def plot_mesh_comparison(clean_v, adv_v, faces, label, clean_pred, adv_pred, save_path=None):
    """
    Visualizes clean vs. adversarial mesh side-by-side.
    """
    faces = faces.astype(int)
    corners_clean = clean_v[faces]  # (F, 3, 3)
    corners_adv = adv_v[faces]
    true_class = class_to_label[label].capitalize()
    pred_class = class_to_label[clean_pred].capitalize()
    adv_class = class_to_label[adv_pred].capitalize()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), subplot_kw={'projection': '3d'}, )
    mesh1 = Poly3DCollection(corners_clean, alpha=0.25, facecolor='lightblue', edgecolor='k')
    axes[0].add_collection3d(mesh1)
    axes[0].set_title(f"Clean Prediction: {pred_class}")
    axes[0].set_box_aspect([1, 1, 1])
    axes[0].set_axis_off()

    mesh2 = Poly3DCollection(corners_adv, alpha=0.25, edgecolor='#670D2F', facecolor='#EF88AD')
    axes[1].add_collection3d(mesh2)
    axes[1].set_title(f"Attack Prediction: {adv_class}")
    axes[1].set_axis_off()
    axes[1].set_box_aspect([1, 1, 1])
    axes[1].set_axis_off()

    for ax, verts in zip([axes[0], axes[1]], [clean_v, adv_v]):
        max_range = (verts.max(axis=0) - verts.min(axis=0)).max() / 2
        mid = verts.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    plt.suptitle(f"Adversarial Mesh (True label = {true_class})", fontsize=16, fontweight='bold')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

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