
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


import numpy as np 

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


import torch

from torch import Tensor
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.functional import normalize
from torch.nn import Sequential, Linear, ReLU
from torch.distributions import Categorical

from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Based on https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pyg/pointnet-classification
# /02_pointnet_plus_plus.ipynb#scrollTo=0a089e90

os.environ['TORCH'] = torch.__version__

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

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

class AddCentralityNodeFeatures(BaseTransform):
    def __init__(self, num_workers=4, sanity_prints=False):
        self.num_workers = num_workers
        self.sanity_prints = sanity_prints

    def __call__(self, data):
        """
        Add as trasnform (on the fly)
        """
        # data.x = self.compute_node_importance_scores(data)
        data.x = None
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
    
def create_dataset(dataset_name, name='10', 
                   num_point_to_sample=2048, k=10,
                   force_reload=False,use_pre_filter=False,
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

    pre_transform = T.Compose([
        T.NormalizeScale(), 
    ])


    # transform = T.Compose([T.NormalizeScale(),
    #                        T.SamplePoints(num=num_point_to_sample, remove_faces=False)])
    #, T.FaceToEdge(remove_faces = False), AddCentralityNodeFeatures()])
    transform = None
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


class SetAbstraction(torch.nn.Module):
    def __init__(self, ratio, ball_query_radius, nn):
        super().__init__()
        self.ratio = ratio
        self.ball_query_radius = ball_query_radius
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.ball_query_radius, batch, batch[idx], max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx] # @ NOTE: remove # if needed 
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSetAbstraction(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNetPlusPlus(torch.nn.Module):
    def __init__(self, set_abstraction_ratio_1, set_abstraction_ratio_2, set_abstraction_radius_1,
                 set_abstraction_radius_2, dropout):
        super().__init__()
        # Input channels account for both `pos` and node features.
        self.sa1_module = SetAbstraction(set_abstraction_ratio_1, set_abstraction_radius_1, MLP([3, 64, 64, 128]))
        self.sa2_module = SetAbstraction(set_abstraction_ratio_2, set_abstraction_radius_2,
                                         MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSetAbstraction(MLP([256 + 3, 256, 512, 1024]))
        self.mlp = MLP([1024, 512, 256, 10], dropout=dropout, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        return self.mlp(x).log_softmax(dim=-1)


def train_step(epoch):
    model.train()
    epoch_loss, correct = 0, 0
    num_train_examples = len(train_loader)

    progress_bar = tqdm(range(num_train_examples),
                        desc=f"Training Epoch {epoch}/{num_epochs}"
                        )
    for batch_idx in progress_bar:
        data = next(iter(train_loader)).to(device)

        optimizer.zero_grad()
        prediction = model(data)
        loss = F.nll_loss(prediction, data.y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct += prediction.max(1)[1].eq(data.y).sum().item()

    epoch_loss = epoch_loss / num_train_examples
    epoch_accuracy = correct / len(train_loader.dataset)


def val_step(epoch):
    model.eval()

    epoch_loss, correct = 0, 0
    num_val_examples = len(val_loader)

    progress_bar = tqdm(range(num_val_examples), desc=f"Validation Epoch {epoch}/{num_epochs}")

    for batch_idx in progress_bar:
        data = next(iter(val_loader)).to(device)

        with torch.no_grad():
            prediction = model(data)

        loss = F.nll_loss(prediction, data.y)
        epoch_loss += loss.item()
        correct += prediction.max(1)[1].eq(data.y).sum().item()

    epoch_loss = epoch_loss / num_val_examples
    epoch_accuracy = correct / len(val_loader.dataset)


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
    plt.title(f'PointNet++ | bumbling_sweep11 \n\n\n')

    plt.legend()

    plt.tight_layout()
    if save_fig:
        # save_path = os.path.join("images/", "point_net_pp_bumbling_sweep11.png")
        plt.savefig("point_net_pp_bumbling_sweep11.png")

def get_filtered_datasets(dataset, max_points=2048):
    print(f"Original dataset: {len(dataset)} samples.")
    filtered_dataset = [data for data in dataset if data.pos.size(0) <= max_points]
    print(f"Filtered dataset (max_points = {max_points}): {len(filtered_dataset)} samples.")
    return filtered_dataset

def train(filter_datasets = False, max_points=2048):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-4
    num_epochs = 25
    num_workers = 4

    batch_size = 8
    num_point_to_sample = 2048
    set_abstraction_ratio_1 = 0.4624
    set_abstraction_ratio_2 = 0.2868
    set_abstraction_radius_1 = 0.7562
    set_abstraction_radius_2 = 0.5
    dropout = 0.18

    pre_transform = T.NormalizeScale()
    # transform = T.SamplePoints(num_point_to_sample, remove_faces=False)

    # train_dataset = ModelNet(root="ModelNet10", name='10', train=True, transform=transform, pre_transform=pre_transform)
    # val_dataset = ModelNet(root="ModelNet10", name='10', train=False, transform=transform, pre_transform=pre_transform)
    train_dataset, val_dataset = create_dataset(
        dataset_name="ModelNet10",
        name='10',
        num_point_to_sample=num_point_to_sample,
        use_pre_filter=False,
        force_reload=False,
        include_train=True,
        include_test=True
    )

    if filter_datasets:
        train_dataset = get_filtered_datasets(train_dataset, max_points=max_points)
        val_dataset = get_filtered_datasets(val_dataset, max_points=max_points)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = PointNetPlusPlus(
        set_abstraction_ratio_1,
        set_abstraction_ratio_2,
        set_abstraction_radius_1,
        set_abstraction_radius_2,
        dropout).to(device)
    
    # model = nn.DataParallel(model)
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
            if filter_datasets:
                file_name = f"filtered_le{max_points}_PointNetPP_epochs{epoch}_val_acc{int(val_epoch_accuracy * 100)}.pt"
            else:
                file_name = f"PointNetPP_epochs{epoch}_val_acc{int(val_epoch_accuracy * 100)}.pt"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, f"checkpoints/{file_name}")

    plot_training_history(num_epochs=num_epochs, train_losses=train_losses, val_losses=val_losses,
                          train_accuracies=train_accuracies, val_accuracies=val_accuracies)

    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, f"checkpoints/{file_name}")

if __name__ == '__main__':
    train(filter_datasets = True, max_points=2048)
