import os
import json
import random
import time
import glob
import tqdm 
from multiprocessing import Pool, cpu_count

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
from torch_geometric.utils import to_networkx, to_dense_adj, dropout, add_self_loops, sort_edge_index

import torch_cluster
import torch

from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.nn import Sequential, Linear, ReLU
from torch.distributions import Categorical

from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from utils.config import default_args, filtered_classes,manual_probs,modelnet10_classes,modelnet10_labels
torch.manual_seed(seed=42)

def generate_single_walk(args):
    def manual_dist_probs(distances_list, relax_consideration_flag, sanity_prints=False):
        distances_list = np.array(distances_list)
        n = distances_list.shape[0]
        scores = manual_probs[n]
        if relax_consideration_flag:
            scores = np.ones(n) - scores
            probs = np.exp(scores) / np.sum(np.exp(scores))
        else:
            probs = np.array(scores)
        return probs

    pc, start_point, walks_len, k, num_walks, relax_frac, use_relax_walk_idx, p_dist, n_workers = args
    
    kd_tree = KDTree(pc)
    walk = np.zeros((walks_len + 1,), dtype=np.int32) - 1

    walk[0] = start_point
    u = walk[0].item()
    total_edges_undirected = set()
    total_edges_directed = []
    relax_walk_step_idx = np.ceil(walks_len * relax_frac) if use_relax_walk_idx else walks_len

    for walk_step in range(1, walks_len + 1):
        current_point = pc[walk[walk_step - 1]]
        distances, neighbors = kd_tree.query(x=current_point, k=20, p=p_dist, workers=n_workers)
        neighbors_to_consider = []
        n_dist = []
        curr_idx = 0
        walk_step_relax_flag = walk_step > relax_walk_step_idx
        for n_idx, neighbor in enumerate(neighbors):
            edge_key = frozenset({u, neighbor})
            if edge_key in total_edges_undirected and not walk_step_relax_flag:
                continue
            neighbors_to_consider.append(neighbor)
            n_dist.append(distances[n_idx])
            curr_idx += 1
            if len(neighbors_to_consider) == 10:
                break
        if curr_idx >= 1:
            probs = manual_dist_probs(n_dist,walk_step_relax_flag)
            # probs = rank_bias_probs(np.array(n_dist), walk_step_relax_flag)
            walk[walk_step] = np.random.choice(neighbors_to_consider, p=probs)
        else:
            walk[walk_step] = np.random.choice(neighbors[k:])

        v = walk[walk_step].item()
        new_edge = (u, v)
        total_edges_directed.append(new_edge)

        new_edge_undirected = frozenset({u, v})
        total_edges_undirected.add(new_edge_undirected)
        u = v

    walks_idx = torch.tensor(walk[:walks_len])
    total_edges_directed = torch.tensor(np.array(total_edges_directed), dtype=torch.long)
    return walks_idx, total_edges_directed




class EdgeIndexTransform(BaseTransform):
    def __call__(self, data):   
        #print(f"data.edge_index: {data.edge_index.T}")
        #data.edge_index = torch.cat(data.edge_index.T.unbind(1),dim=0).T
        #print(f"data.edge_index: {data.edge_index}")
        data.edge_index = data.edge_index.T
        edge_index, _ = add_self_loops(data.edge_index)
        edge_index = sort_edge_index(edge_index)
        # print(edge_index)
        #  data.edge_index = edge_index
        g = to_networkx(data, to_undirected=True)
        adj = to_dense_adj(edge_index, max_num_nodes=data.num_nodes)[0].float()
        transition_matrix = F.normalize(adj, p=1, dim=1)

        data.g = g
        data.adj = adj.float()
        data.Prob = transition_matrix
        data.edge_index = edge_index
        data.pos = data.pos.float()
        
        return data


class AddCentralityNodeFeatures(BaseTransform):
    def __init__(self, num_workers=default_args["num_workers"], sanity_prints=False):
        self.num_workers = num_workers
        self.sanity_prints = sanity_prints

    def __call__(self, data):
        """
        Add as trasnform (on the fly)
        """
        data.x = self.compute_node_importance_scores(data)
        return data

    def compute_node_importance_scores(self, pc_data, calc_divided=False, add_info_centrality=False,
                                       add_percolation_centrality=True):
        """
        Compute centrality and structural measures and stores it as node features.
        Convert PyG graph to NetworkX for easier centrality calculations

        degree = torch.from_numpy(np.array([val for _, val in G.degree()]))
        --> degree: tensor([ 0,  0,  2,  ...,  4,  4, 22])

        pagerank = torch.from_numpy(np.array(list(nx.pagerank(G).values())))
        --> pagerank: tensor([9.1463e-05, 9.1463e-05, 4.4529e-04,  ..., 4.3281e-04, 4.3184e-04,1.2675e-03], dtype=torch.float64)

        features = torch.stack([degree,pagerank,clustering, betweenness,closeness], axis=1) # stucking axis=1 make each row i to contain the measures each node i.
        --> features: tensor([[0.0000e+00, 9.1463e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                    [0.0000e+00, 9.1463e-05, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                                    [2.0000e+00, 4.4529e-04, 0.0000e+00, 8.5936e-04, 2.8449e-02],
                                    ...,
                                    [4.0000e+00, 4.3281e-04, 3.3333e-01, 3.9461e-04, 2.6414e-02],
                                    [4.0000e+00, 4.3184e-04, 5.0000e-01, 1.8781e-04, 3.9624e-02],
                                    [2.2000e+01, 1.2675e-03, 4.1579e-01, 4.7221e-03, 3.1103e-02]],
                                       dtype=torch.float64)

        features = torch.nn.functional.normalize(features)
        --> features: tensor([[2.1057e-05, 3.2018e-04, 1.0000e+00,  ..., 0.0000e+00, 7.6703e-06,1.7271e-06],
                                    [8.1504e-05, 3.0428e-04, 9.9996e-01,  ..., 9.2688e-03, 2.2976e-05,4.0525e-05],
                                    [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00, 1.0000e+00,0.0000e+00],
                                    ...,
                                    [3.7700e-05, 3.4016e-04, 1.0000e+00,  ..., 2.4116e-03, 9.3001e-06,1.0138e-05],
                                    [5.0755e-05, 3.6780e-04, 9.9998e-01,  ..., 6.0297e-03, 1.0823e-05,1.5205e-06],
                                    [1.1574e-05, 4.1444e-04, 1.0000e+00,  ..., 0.0000e+00, 5.6553e-06,2.0249e-06]],
                                    dtype=torch.float64)

        :param pc_data: Data object
        :param calc_divided: (for aesthetics code)
        :param add_info_centrality: G needs to be connected (Currently False)
        :param add_percolation_centrality: Need to reed about
        """
        G = to_networkx(pc_data, to_undirected=True)
        calculated_measures = []

        if calc_divided:
            degree = torch.from_numpy(np.array(list(nx.degree_centrality(G).values())))
            closeness = torch.from_numpy(np.array(list(nx.closeness_centrality(G).values())))
            harmonic = torch.from_numpy(np.array(list(nx.harmonic_centrality(G).values())))
            betweenness = torch.from_numpy(np.array(list(nx.betweenness_centrality(G).values())))
            clustering = torch.from_numpy(np.array(list(nx.clustering(G).values())))
            pagerank = torch.from_numpy(np.array(list(nx.pagerank(G).values())))
            calculated_measures = [degree, closeness, harmonic, betweenness, clustering, pagerank]
        else:
            centrality_measures_nx = [nx.degree_centrality, nx.closeness_centrality, nx.harmonic_centrality,
                                      nx.betweenness_centrality, ]  # nx.clustering, nx.pagerank]
            for measure in centrality_measures_nx:
                calculated_measures.append(torch.from_numpy(np.array(list(measure(G).values()))))

        if add_info_centrality:
            information = torch.from_numpy(np.array(list(nx.information_centrality(G).values())))
            calculated_measures.append(information)
        if add_percolation_centrality:
            percolation = torch.from_numpy(np.array(list(nx.percolation_centrality(G).values())))
            calculated_measures.append(percolation)

        features = torch.stack(calculated_measures, axis=1)

        if self.sanity_prints:
            print(f"Stacked features: {features}")
            print(f"Normalize features: {features}")
        features = torch.nn.functional.normalize(features)
        features = features.float()
        #features = torch.tensor(features, dtype=torch.float)

        return features
    
class IncludeMeshFeatures(BaseTransform):
    def __init__(self, num_points=2048):
        self.num_points = num_points

    def filter_faces(self, data_face):
        valid_mask = (data_face < self.num_points).all(dim=0)
        data_face[:, valid_mask]
        return filtered_face
        
    def filter_faces_by_indices(self, data_face, indices):
        mask = torch.isin(data_face, indices).all(dim=0)
        return data_face[:, mask]
    
    def __call__(self, data):
        transform = T.SamplePoints(self.num_points,remove_faces=False)
        data = transform(data)
        data.face = filtered_faces
        return data
    
class AddWalkFeature(BaseTransform):
    def __init__(self, n_sample=default_args["n_sample"], num_walks=default_args["num_walks"],
                 walks_len=default_args["walks_len"], relax_frac=0.8, num_workers=default_args["num_workers"],
                 k=default_args["k"],
                 sanity_prints=False):
        self.num_walks = num_walks
        self.walks_len = walks_len
        self.k = k
        self.p = np.inf
        self.relax_frac = relax_frac
        self.num_workers = num_workers
        self.sanity_prints = sanity_prints

    def __call__(self, data):
        pc = data.pos.cpu().numpy()
        walks, walks_idx, edge_index = self.parallel_generate_walks(pc)
        data.walks = walks
        data.walks_idx = walks_idx
        data.edge_index = edge_index
        return data

    def prev_call(self, data):
        pc = data.pos
        walks, total_walks_idx, edge_index, edge_attr = self.generate_random_walks(pc)
        data.walks = walks
        data.walks_idx = total_walks_idx
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data

    def parallel_generate_walks(self, pc, use_relax=True):
        n_vertices = pc.shape[0]

        # unique_start_points = np.random.choice(n_vertices, size=self.num_walks, replace=False)

        points_tensor = torch.tensor(pc, dtype=torch.float32)


        ratio = self.num_walks / points_tensor.size(0)
        sampled_indices = fps(points_tensor, ratio=ratio, random_start=True)
        # sampled_points = points_tensor[sampled_indices]
        walk_args = [
            (pc, start, self.walks_len, self.k, self.num_walks, self.relax_frac, use_relax, self.p, self.num_workers)
            for start in sampled_indices
        ]

        with Pool(min(cpu_count(), 8)) as pool:
            results = pool.map(generate_single_walk, walk_args)

        walks_idx, edge_indices = zip(*results)
        # print(walks_idx, edge_indices, sep='\n\n')
        walks = torch.stack([torch.tensor(pc[walk_idx]) for walk_idx in walks_idx])
        walks_idx = torch.stack(walks_idx)
        edge_index = torch.tensor(torch.stack(edge_indices), dtype=torch.long).T
        edge_index = torch.cat(edge_indices,0)

        return walks, walks_idx, edge_index

    def generate_random_walks(self, pc, relax_frac=0.8, use_relax_walk_idx=False):
        def manual_dist_probs(distances_list, relax_consideration_flag):
            distances_list = np.array(distances_list)
            n = distances_list.shape[0]
            scores = manual_probs[n]

            if relax_consideration_flag:
                scores = np.ones(n) - scores
                probs = np.exp(scores) / np.sum(np.exp(scores))
            else:
                probs = np.array(scores)
            return probs

        kd_tree = KDTree(pc)
        n_vertices = pc.shape[0]

        total_walks = []
        # edges_attr = []
        # total_edges = []
        total_edges_undirected = set()
        total_walks_idx = []
        edges_attr = np.zeros((self.num_walks, self.walks_len))

        optional_starts = list(range(0, n_vertices))
        visited = np.zeros((n_vertices + 1,), dtype=bool)

        relax_walk_idx = np.ceil(self.num_walks * relax_frac)
        relax_walk_step_idx = np.ceil(self.walks_len * relax_frac)

        for i in range(self.num_walks):
            start_point = np.random.choice(optional_starts)

            visited[start_point] = True
            optional_starts.remove(start_point)

            walk = np.zeros((self.walks_len + 1,), dtype=np.int32) - 1
            walk[0] = start_point

            u = walk[0].item()
            walk_idx_relax_flag = use_relax_walk_idx & (i > relax_walk_idx)

            for walk_step in range(1, self.walks_len + 1):
                current_point = pc[walk[walk_step - 1]]
                distances, neighbors = kd_tree.query(x=current_point, k= 20, p=self.p, workers=self.num_workers)
                neighbors_to_consider = []
                n_dist = []
                curr_idx = 0
                walk_step_relax_flag = walk_idx_relax_flag & (walk_step > relax_walk_step_idx)
                for n_idx, neighbor in enumerate(neighbors):
                    # u_neighbor_rule = (u, neighbor) not in total_edges
                    u_neighbor_rule = frozenset({u, neighbor}) not in total_edges_undirected
                    if not u_neighbor_rule or not walk_step_relax_flag:
                        continue
                    # strict_consideration_rule = (neighbor, u) not in total_edges
                    # relax_consideration_flag = walk_step_relax_flag
                    # if strict_consideration_rule | relax_consideration_flag:
                    neighbors_to_consider.append(neighbor)
                    n_dist.append(distances[n_idx])

                    if len(n_dist) == 10:
                        break
                if len(n_dist) > 1:
                    probs = manual_dist_probs(n_dist, relax_consideration_flag)
                    walk[walk_step] = np.random.choice(neighbors_to_consider, p=probs)
                else:
                    walk[walk_step] = np.random.choice(neighbors[self.k:])

                v = walk[walk_step].item()
                new_edge = (u, v)
                total_edges_directed.append(new_edge)
                new_edge = frozenset({u, v})
                total_edges_undirected.add(new_edge)

                # attr = np.zeros((self.num_walks,))
                #  attr[i] = 1
                # edges_attr.append(attr)
                edges_attr[i, walk_step] = 1
                u = v

            # for walk_step in range(1, self.walks_len + 1):
            #     visited[walk[walk_step]] = True

            # walks_idx = torch.tensor(walk[:self.walks_len])
            # walk_pcs = pc[walks_idx]
            # total_walks_idx.append(walks_idx)
            walks_idx = torch.tensor(walk[:self.walks_len])
            total_walks_idx[i, :] = walks_idx
            total_walks[i, :] = pc[walks_idx]
            # total_walks.append(walk_pcs)

        # total_walks_idx = np.array(total_walks_idx)
        edges_attr = np.array(edges_attr)
        total_walks = torch.stack(total_walks)
        edges_attributes = torch.tensor(edges_attr, dtype=torch.long)
        total_edges = torch.tensor(total_edges_directed, dtype=torch.long).T
        return total_walks, total_walks_idx, total_edges, edges_attributes

    def generate_hard_random_walks(self, pc):
        kd_tree = KDTree(pc)
        n_vertices = pc.shape[0]

        total_walks = []
        edges_attr = []
        edges_check = set()  # []
        total_edges = []
        total_walks_idx = []

        optional_starts = list(range(0, n_vertices))
        visited = np.zeros((n_vertices + 1,), dtype=bool)

        for i in range(self.num_walks):
            start_point = np.random.choice(optional_starts)

            visited[start_point] = True
            optional_starts.remove(start_point)

            walk = np.zeros((self.walks_len + 1,), dtype=np.int32) - 1
            walk[0] = start_point

            u = walk[0].item()
            for walk_step in range(1, self.walks_len + 1):
                current_point = pc[walk[walk_step - 1]]
                distances, neighbors = kd_tree.query(x=current_point, k=self.k * 2, workers=self.num_workers)
                neighbors = kd_tree.query_ball_point(current_point, r=0.5, p=2, return_sorted=True, )
                neighbors_to_consider = []
                n_dist = []
                for n_idx, neighbor in enumerate(neighbors):
                    if (u, neighbor) not in total_edges and (neighbor, u) not in total_edges:
                        neighbors_to_consider.append(neighbor)
                        n_dist.append(distances[n_idx])
                    if len(neighbors_to_consider) == self.k:
                        break
                if len(neighbors_to_consider) > 1:
                    probs = rank_bias_probs(n_dist)
                    walk[walk_step] = np.random.choice(neighbors_to_consider, p=probs)
                else:
                    walk[walk_step] = np.random.choice(neighbors[self.k:])

                v = walk[walk_step].item()
                new_edge = (u, v)
                total_edges.append(new_edge)

                attr = np.zeros((self.num_walks,))
                attr[i] = 1
                edges_attr.append(attr)
                u = v
            for walk_step in range(1, self.walks_len + 1):
                visited[walk[walk_step]] = True
            walks_idx = torch.tensor(walk[:self.walks_len])
            walk_pcs = pc[walks_idx]
            total_walks_idx.append(walks_idx)
            total_walks.append(walk_pcs)

        total_walks_idx = np.array(total_walks_idx)
        edges_attr = np.array(edges_attr)
        total_walks = torch.stack((total_walks))
        edges_attributes = torch.tensor(edges_attr, dtype=torch.long)
        total_edges = torch.tensor(total_edges, dtype=torch.long).T
        return total_walks, total_walks_idx, total_edges, edges_attributes
