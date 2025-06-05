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

from glob import glob
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
import os

from concurrent.futures import ThreadPoolExecutor
import numpy as np 

import networkx as nx
import trimesh
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
from torch_geometric.transforms import LocalDegreeProfile, RemoveDuplicatedEdges
from torch_geometric.utils import degree

from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


default_args = {
    "k": 10,
    "n_sample": 2048,
    "num_walks": 32,  
    "walks_len": 128,
    "batch_size": 16,
    "num_workers": 4  
}


class MeshEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        edge_feat = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.mlp(edge_feat)

class MeshCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = MeshEdgeConv(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        return F.relu(x)

class MeshCNNClassifier(nn.Module):
    def __init__(self, in_channels=6, num_classes=10):
        super().__init__()
        self.block1 = MeshCNNBlock(in_channels, 64)
        self.block2 = MeshCNNBlock(64, 128)
        self.block3 = MeshCNNBlock(128, 256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.block1(x, edge_index)
        x = self.block2(x, edge_index)
        x = self.block3(x, edge_index)
        x = global_max_pool(x, data.batch)
        return self.classifier(x)


os.environ['TORCH'] = torch.__version__
torch.manual_seed(seed=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelnet10_labels = {
    'bathtub': 0,
    'bed': 1,
    'chair': 2,
    'desk': 3,
    'dresser': 4,
    'monitor': 5,
    'night_stand': 6,
    'sofa': 7,
    'table': 8,
    'toilet': 9,
}

modelnet10_classes = {v: k for k, v in modelnet10_labels.items()}

class MyPointsSampling(BaseTransform):
    r"""Uniformly samples a fixed number of points on the mesh faces according
    to their face area (functional name: :obj:`sample_points`).

    Args:
        num (int): The number of points to sample.
        remove_faces (bool, optional): If set to :obj:`False`, the face tensor
            will not be removed. (default: :obj:`True`)
        include_normals (bool, optional): If set to :obj:`True`, then compute
            normals for each sampled point. (default: :obj:`False`)
    """
    def __init__(
        self,
        num: int = 2048,
        w: float = 0.5,
        remove_faces: bool = False,
        include_normals: bool = False,
    ):
        self.num = num
        self.remove_faces = remove_faces
        self.include_normals = include_normals
        self.w = w
    

    def __call__(self, data):
        # print(f"Previous data:\n{data}")
        full_verts = data.pos.cpu().numpy()
        full_faces = data.face.T.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=full_verts, faces=full_faces, process=False)
        new_pos , new_face_indices = mesh.sample(count=self.num, return_index =True) # trimesh.sample.sample_surface(mesh,count=self.num)
        new_pos = torch.from_numpy(new_pos).to(torch.float)
        new_face = full_faces[new_face_indices,:]
        new_face = torch.from_numpy(new_face).t().contiguous()
        new_data = Data(pos=new_pos, face=new_face, y=data.y.clone().detach(), orig_pos=data.pos.clone().detach())
        return new_data
        
    def __call__(self, data):
        data.transform = T.SamplePoints(self.num, remove_faces=False)
        full_verts = data.pos.cpu().numpy()
        full_faces = data.face.T.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=full_verts, faces=full_faces, process=False)
        sampled_xyz, face_indices = mesh.sample(count=self.num, return_index=True)

        sampled_faces = full_faces[face_indices]  # [N, 3]
        sampled_vertex_ids = np.unique(sampled_faces.flatten())
        old_to_new = {old: new for new, old in enumerate(sampled_vertex_ids)}
        remapped_faces = np.vectorize(old_to_new.get)(sampled_faces)

        #new_pos = torch.tensor(full_verts[sampled_vertex_ids], dtype=torch.float)
        face_tensor = torch.tensor(remapped_faces.T, dtype=torch.long)
        edge_index = torch.cat([
                    face_tensor[:2],
                    face_tensor[1:],
                    face_tensor[::2],], dim=1)
        num_nodes = data.pos.shape[0]
        edge_index = torch_geometric.utils.to_undirected(edge_index, num_nodes=num_nodes)
        
        return Data(
            pos= data.pos,
            edge_index=edge_index,
            face=face_tensor,
            y=data.y.clone(),
            orig_pos=data.pos.clone()
        )
    
    def align_faces(self, data, sampled_indices, invert=False):
        full_verts = data.pos.cpu().numpy()
        full_faces = data.face.T.cpu().numpy()
    
        mesh = trimesh.Trimesh(vertices=full_verts, faces=full_faces, process=False)
    
        v_faces = mesh.vertex_faces[sampled_indices]  
        candidate_face_ids = np.unique(v_faces[v_faces != -1]) 
    
        face_subset = mesh.faces[candidate_face_ids]
        mask = np.any(np.isin(face_subset, sampled_indices, invert), axis=1)
    
        selected_faces = face_subset[mask]
        
        
        return selected_faces
    def trimesh_sample(self,data):
        full_verts = data.pos.cpu().numpy()
        full_faces = data.face.T.cpu().numpy()
        mesh = trimesh.Trimesh(vertices=full_verts, faces=full_faces, process=False)
        sample = trimesh.sample.sample_surface_even(mesh,count=self.num)
        new_pos = sample.samples 
        # new_pos = data.pos[sampled_indices,:]
        new_faces = full_faces[sample.face_index]
        data.pos = torch.tensor(new_pos, dtype=torch.float)
        data.face = torch.tensor(new_faces.numpy())
        return data
    
    def weighted_sample(self, data, lower_sampling=True, invert=False):
        full_verts = data.pos.cpu().numpy()
        full_faces = data.face.T.cpu().numpy()
        full_num_points = full_verts.shape[0]
        
        np_faces =  data.face.T.cpu().numpy()
        faces_df = pd.DataFrame.from_records(np_faces)
        verts_counts = pd.concat([faces_df[col].value_counts() for col in faces_df.columns], axis=1).fillna(0).sum(axis=1).astype(int).sort_values(ascending=False)
        # data.verts_counts = verts_counts
        
        indices = verts_counts.index.to_numpy()
        weights = verts_counts.to_numpy() ** self.w
        probabilities = weights / weights.sum()
        if lower_sampling:
            probabilities = 1 - probabilities
            probabilities = probabilities / probabilities.sum()
            
        resample_points = self.num > indices.shape[0]
        sampled_indices = np.random.choice(indices, size=self.num, replace = resample_points, p=probabilities)
        # sampled_value_counts = verts_counts[sampled_indices].sort_values(ascending=False)
        
        # pos_indices = np.arange(full_num_points)
        # not_sampled_indices = np.setdiff1d(pos_indices, sampled_indices, assume_unique=False)
        # not_in_new_pos = data.pos[not_sampled_indices, :]
        # data.pos = new_pos
        # selected_faces = self.align_faces(data, sampled_indices)
        # data.face = selected_faces


        new_pos = data.pos[sampled_indices,:]
        mesh = trimesh.Trimesh(vertices=full_verts, faces=full_faces, process=False)
    
        v_faces = mesh.vertex_faces[sampled_indices]  
        candidate_face_ids = np.unique(v_faces[v_faces != -1]) 
    
        face_subset = mesh.faces[candidate_face_ids]
        mask = np.any(np.isin(face_subset, sampled_indices, invert=False), axis=1)
    
        selected_faces = face_subset[mask]
        data.pos = new_pos
        data.face = selected_faces
        return data
        
    def low_high_sample(self,data):
        np_faces =  data.face.T.numpy()
        faces_df = pd.DataFrame.from_records(np_faces)
        vertex_counts = pd.concat([faces_df[col].value_counts() for col in faces_df.columns], axis=1).fillna(0).sum(axis=1).astype(int).sort_values(ascending=False)
        req_quantiles = [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]
        quantiles_map = {q: q_val for q, q_val in zip(req_quantiles, np.quantile(vertex_counts,req_quantiles))}
    
        lower_chosen_threshold = self.find_lower_threshold(vertex_counts,quantiles_map,n_points=self.num)
        upper_chosen_threshold = self.find_upper_threshold(vertex_counts, quantiles_map,n_points=self.num)
        
        lowest_verts  = vertex_counts[vertex_counts < lower_chosen_threshold]
        lowest_indices_sample = lowest_verts.sample(min(lowest_verts.shape[0],self.num)).index.to_numpy()
    
        highest_verts = vertex_counts[vertex_counts > upper_chosen_threshold]
        highest_indices_sample = highest_verts.sample(min(highest_verts.shape[0],self.num)).index.to_numpy()
        
        
        
    def find_lower_threshold(self,vertex_counts, quantiles_map, prints=False):
        for q, threshold in quantiles_map.items():
            count_above = (vertex_counts <= threshold).sum()
            if count_above >= self.num:
                chosen_threshold = threshold
                if prints:
                    print(f"Quantile: {q}, Lower Threshold: {threshold}, Count: {count_above}")
                return chosen_threshold
    
    
    def find_upper_threshold(self,vertex_counts, quantiles_map, prints= False):
        sorted_quantiles_by_val = dict(sorted(quantiles_map.items(), key=lambda x: x[1], reverse=True))
        for q, threshold in sorted_quantiles_by_val.items():
            count_above = (vertex_counts >= threshold).sum()
            if count_above >=  self.num:
                chosen_threshold = threshold
                if prints:
                    print(f"Quantile: {q}, Upper Threshold: {threshold}, Count: {count_above}")
                return chosen_threshold
            
    
    def plot_mesh(self,data, quantiles_map):
        """
        Visualizes a 3D mesh from a torch_geometric.data.Data object using matplotlib.
    
        Args:
            data: A torch_geometric.data.Data object with `pos` (N, 3) and `face` (3, F)
        """
        verts = data.pos.cpu().numpy()
        faces = data.face.T.cpu().numpy()  # (F, 3)
    
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(verts[faces], alpha=0.7, linewidth=0.3)
    
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
    
        max_range = (verts.max(axis=0) - verts.min(axis=0)).max() / 2.0
        mid = verts.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
        ax.set_box_aspect([1, 1, 1])
        plt.tight_layout()
        plt.show()
    
    def plot_pc_high_low(self, data, lowest_indices_sample,highest_indices_sample):
        pos_highest = data.pos[highest_indices_sample,:]
        pos_lowest = data.pos[lowest_indices_sample,:]
        
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos_highest[:, 0], pos_highest[:, 1], pos_highest[:, 2],edgecolor='#DD5746', s=50, alpha=.5,c='#DD5746',linewidth=1)
        ax.scatter(pos_lowest[:, 0], pos_lowest[:, 1], pos_lowest[:, 2],edgecolor='#4793AF', s=50, alpha=.5,c='#4793AF',linewidth=1)
        plt.show()
    
    def plot_in_VS_out(self,data,sampled_indices,sampling_method=""):
                
        new_pos = data.pos.cpu().numpy()
        not_in_new_pos = data.not_pos.cpu().numpy()
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(new_pos[:, 0], new_pos[:, 1], new_pos[:, 2],edgecolor='#DD5746', s=50, alpha=.5,c='#DD5746',linewidth=1)
        ax.scatter(not_in_new_pos[:, 0], not_in_new_pos[:, 1], not_in_new_pos[:, 2],edgecolor='#4793AF', s=50, alpha=.5,c='#4793AF',linewidth=1)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])   
        
        max_range = (new_pos.max(axis=0) - new_pos.min(axis=0)).max() / 2.0
        mid = new_pos.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
        ax.set_box_aspect([1, 1, 1])
        plt.tight_layout()
            
        plt.suptitle(f"{sampling_method} Sampling",fontweight='bold')
        plt.show()
    
    
    def get_face_stats(self,data):
        print(data.vertex_count.describe())
    
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num})'
    
class AddCentralityNodeFeatures(BaseTransform):
    def __init__(self, num_workers=4, sanity_prints=False):
        self.num_workers = num_workers
        self.sanity_prints = sanity_prints
        self._cache = {}

    def __call__(self, data):
        """
        Add as transform (on the fly)
        """
        data.x = self.compute_node_importance_scores(data)
        return data
    
    def not_parallel_compute_node_importance_scores(self, pc_data, calc_divided=False, add_info_centrality=False,
                                       add_percolation_centrality=True):
        G = to_networkx(pc_data, to_undirected=True)


        degree = torch.from_numpy(np.array(list(nx.degree_centrality(G).values())))
        closeness = torch.from_numpy(np.array(list(nx.closeness_centrality(G).values())))
        harmonic = torch.from_numpy(np.array(list(nx.harmonic_centrality(G).values())))
        betweenness = torch.from_numpy(np.array(list(nx.betweenness_centrality(G).values())))
        clustering = torch.from_numpy(np.array(list(nx.clustering(G).values())))
        pagerank = torch.from_numpy(np.array(list(nx.pagerank(G).values())))
        percolation = torch.from_numpy(np.array(list(nx.percolation_centrality(G).values())))
        calculated_measures = [degree, closeness, harmonic, betweenness, clustering, pagerank, percolation]
        
        if add_info_centrality:
            information = torch.from_numpy(np.array(list(nx.information_centrality(G).values())))
            calculated_measures.append(information)

        features = torch.stack(calculated_measures, axis=1)

        if self.sanity_prints:
            print(f"Stacked features: {features}")
            print(f"Normalize features: {features}")
        features = torch.nn.functional.normalize(features)
        features = features.float()
        #features = torch.tensor(features, dtype=torch.float)

        return features

    def compute_node_importance_scores(self, pc_data, calc_divided=True, add_info_centrality=False):
        """
        Compute centrality and structural measures and stores it as node features.
        Uses parallel processing and caching for better performance.
        """

        cache_key = hash(pc_data.edge_index.cpu().numpy().tobytes())
        if cache_key in self._cache:
            return self._cache[cache_key]

        G = to_networkx(pc_data, to_undirected=True)
        

        def compute_degree_centrality():
            torch_geometric.utils.degree
            return torch.from_numpy(np.array(list(nx.degree_centrality(G).values())))
        
        def compute_closeness_centrality():
            return torch.from_numpy(np.array(list(nx.closeness_centrality(G).values())))
        
        def compute_harmonic_centrality():
            return torch.from_numpy(np.array(list(nx.harmonic_centrality(G).values())))
        
        def compute_betweenness_centrality():
            return torch.from_numpy(np.array(list(nx.betweenness_centrality(G).values())))
        
        def compute_clustering():
            return torch.from_numpy(np.array(list(nx.clustering(G).values())))
        
        def compute_pagerank():
            return torch.from_numpy(np.array(list(nx.pagerank(G).values())))
        
        def compute_percolation():
            return torch.from_numpy(np.array(list(nx.percolation_centrality(G).values())))


        # full_centrality_functions = [
        #     compute_degree_centrality,
        #     compute_closeness_centrality,
        #     compute_harmonic_centrality,
        #     compute_betweenness_centrality,
        #     compute_clustering,
        #     compute_pagerank,
        #     compute_percolation
        # ]

        centrality_functions = [
            compute_degree_centrality,
            compute_harmonic_centrality,
            compute_percolation
        ]




        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            calculated_measures = list(executor.map(lambda f: f(), centrality_functions))

        if add_info_centrality:
            information = torch.from_numpy(np.array(list(nx.information_centrality(G).values())))
            calculated_measures.append(information)

        features = torch.stack(calculated_measures, axis=1)
        features = torch.nn.functional.normalize(features)
        features = features.float()

 
        self._cache[cache_key] = features
        
        if self.sanity_prints:
            print(f"Stacked features: {features}")
            print(f"Normalize features: {features}")

        return features
 

def load_from_hf():
    HF_TOKEN = "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"
    api = HfApi(token=HF_TOKEN)
    
    repo_id = "talpe/modelnet10-mesh" 
    filenames = ["modelnet10_train.pt", "modelnet10_test.pt"]

    datasets = []
    for fname in filenames:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            repo_type="dataset"
        )
        datasets.append(torch.load(path))

    train_dataset, test_dataset = datasets
    return train_dataset, test_dataset
  
def create_dataset(dataset_name, name='10', 
                   num_point_to_sample=2048, k=10,
                   force_reload=False,use_pre_filter=True,
                   include_train=False,include_test=False):
    """
    The input pipeline PC classification task is not much different from a standard PyTorch-based input pipeline.
    However, we would be using PyTorch Geometrics dataset API for loading the ModelNet datasets which makes loading
    and processing these datasets way easier for us.
    :return:
    """
    def pre_filter_f(data):
        return data.y.item() in [1, 2, 7, 9]
    
    dataset_name = f"ModelNet{name}" if dataset_name is None else dataset_name
    pre_transform = T.Compose([T.NormalizeScale(), MyPointsSampling(num=num_point_to_sample, remove_faces=False)])

    transform = T.Compose([T.FaceToEdge(remove_faces = False), 
                           T.RemoveDuplicatedEdges(),     
                           T.LocalDegreeProfile()])

    # pre_transform = T.Compose([
    #     T.NormalizeScale(), 
    # ])

    #transform = T.Compose([T.SamplePoints(num=num_point_to_sample, remove_faces=False), T.FaceToEdge(remove_faces = False), AddCentralityNodeFeatures()])
    
    pre_filter_func = pre_filter_f if use_pre_filter else None
    
    train_dataset_,test_dataset_ = None, None

    if include_train:
        print("Loading train dataset...")
        train_dataset_ = ModelNet(root=dataset_name, name=name, train=True, transform=transform,
                                pre_transform=pre_transform, force_reload=force_reload, pre_filter=pre_filter_func)
        print(f"Train dataset loaded with {len(train_dataset_)} samples")
    
    if include_test:
        print("Loading test dataset...")
        test_dataset_ = ModelNet(root=dataset_name, name=name, train=False, transform=transform,
                             pre_transform=pre_transform, force_reload=force_reload, pre_filter=pre_filter_func)
        print(f"Test dataset loaded with {len(test_dataset_)} samples")
    return train_dataset_,test_dataset_




def plot_training_history(num_epochs, train_losses, val_losses, train_accuracies, val_accuracies, save_fig=True):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(16, 5))

    n_rows, n_cols, index_pos = 1, 2, 1
    val_color = "#C55300"  # "#820000"  # "#8B322C"
    train_color = "#0A516D"  # "#102C57"  # "#102C57"
    plt.subplot(n_rows, n_cols, index_pos)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color=train_color, linewidth=2, alpha=1.)
    plt.scatter(epochs[::5], train_accuracies[::5], color=train_color, linewidth=0.1, alpha=0.5)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color=val_color, linewidth=2, alpha=1.)
    plt.scatter(epochs[::5], val_accuracies[::5], color=val_color, linewidth=0.1, alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'\n\n\n')  # Accuracy Over Epochs', fontweight='bold')
    plt.legend()

    plt.subplot(n_rows, n_cols, 2)
    plt.plot(epochs, train_losses, label='Train Loss', color=train_color, linewidth=2, alpha=1.)
    plt.plot(epochs, val_losses, label='Validation Loss', color=val_color, linewidth=2, alpha=1.)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'MeshCNN | Node Centrality Features \n\n\n')

    plt.legend()

    plt.tight_layout()
    if save_fig:
        # save_path = os.path.join("images/", "point_net_pp_bumbling_sweep11.png")
        plt.savefig("mesh_cnn_train.png")



def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-4
    num_epochs = 1 # 10
    num_workers = 3 # 4

    batch_size = 1 #8
    num_point_to_sample = 2048

    # train_dataset, val_dataset = create_dataset(
    #     dataset_name="ModelNet10",
    #     name='10',
    #     num_point_to_sample=num_point_to_sample,
    #     use_pre_filter=True,
    #     force_reload=True,
    #     include_train=True,
    #     include_test=True,
    # )
    train_dataset, val_dataset = load_from_hf()
    print(train_dataset[0])
    print(train_dataset[0].pos)
    print(train_dataset[0].face)
    in_channels = min(train_dataset[0].x.shape[0],train_dataset[0].x.shape[1])
    print(f"In channels: {in_channels}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = MeshCNNClassifier(in_channels=in_channels, num_classes=10).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    for epoch in range(1, num_epochs + 1):

        ################################
        # train_step(epoch)
        ################################

        model.train()
        epoch_loss, correct = 0, 0
        num_train_examples = len(train_loader)

        # progress_bar = tqdm(range(num_train_examples), desc=f"Training Epoch {epoch}/{num_epochs}")
        batch_idx = 0
        # for batch_idx in progress_bar
        for data in tqdm(train_loader, desc="Training progress"):
            #data = next(iter(train_loader)).to(device)
            data = data.to(device)
            optimizer.zero_grad()
            prediction = model(data)
            loss = F.nll_loss(prediction, data.y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # correct += prediction.max(1)[1].eq(data.y).sum().item()
            correct += (prediction.argmax(dim=1) == data.y).sum().item()

            if batch_idx % 100 == 0:
                p = (prediction.argmax(dim=1) == data.y).sum().item() / len(data.y)
                print(f"Batch {batch_idx}: Train loss = {loss.item()} | Train corrects = {p:.2%}")
            batch_idx += 1
        train_epoch_loss = epoch_loss / num_train_examples
        train_epoch_accuracy = correct / len(train_loader.dataset)
        train_losses.append(train_epoch_loss)
        train_accuracies.append(train_epoch_accuracy)
        print("===========================================================================")
        print(
            f"Training Epoch {epoch} / {num_epochs}:\n Train loss = {train_epoch_loss} | Train accuracy = {train_epoch_accuracy:.2%}")
        print("===========================================================================")
        ################################
        # val_step(epoch)
        ################################

        model.eval()

        epoch_loss, correct = 0, 0
        num_val_examples = len(val_loader)

        #progress_bar = tqdm(range(num_val_examples), desc=f"Validation Epoch {epoch}/{num_epochs}")

        #for batch_idx in progress_bar:
        for data in tqdm(val_loader, desc="Validation progress"):
            #data = next(iter(val_loader)).to(device)
            data = data.to(device)
            with torch.no_grad():
                logits = model(data)
                pred = logits.argmax(dim=1)
            loss = F.nll_loss(logits, data.y)
            epoch_loss += loss.item()
            correct += (pred == data.y).sum().item()

        val_epoch_loss = epoch_loss / num_val_examples
        val_epoch_accuracy = correct / len(val_loader.dataset)

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        print("===========================================================================")
        print(
            f"Validation Epoch {epoch} / {num_epochs}:\n Val loss = {val_epoch_loss} | Val accuracy = {val_epoch_accuracy:.2%}")
        print("===========================================================================")

        save_checkpoint = val_epoch_accuracy >= 0.75 or epoch == num_epochs
        if save_checkpoint:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                        #'optimizer_state_dict': optimizer.state_dict()},
                       f"victim_models/checkpoints/mesh_cnn_epochs{epoch}_{int(val_epoch_accuracy * 100)}val_acc.pt")

    plot_training_history(num_epochs=num_epochs, train_losses=train_losses, val_losses=val_losses,
                          train_accuracies=train_accuracies, val_accuracies=val_accuracies)

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                f"victim_models/checkpoints/mesh_cnn_epochs{epoch}_{int(val_epoch_accuracy * 100)}val_acc.pt")

if __name__ == '__main__':
    def pre_filter_func(data):
        return data.y.item() in [1, 2, 7, 9]
    train()
