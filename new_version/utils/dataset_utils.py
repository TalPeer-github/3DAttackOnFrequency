import os
import json
import random
import time
import glob
import tqdm 
from collections import defaultdict

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

from utils.config import default_args, filtered_classes,manual_probs
from utils.transforms import AddCentralityNodeFeatures, AddWalkFeature,EdgeIndexTransform,IncludeMeshFeatures
from utils.visalizations import visualize_3d_centralities
from utils.config import default_args, filtered_classes,manual_probs,modelnet10_classes,modelnet10_labels
torch.manual_seed(seed=42)


def pre_filter_f(data):
    return data.y.item() in filtered_classes

def create_dataset_mesh(dataset_name, pre_transform, transform, name='10', num_point_to_sample=2048, force_reload=False,
                        use_pre_filter=True,include_train=False,include_test=False):
    
    dataset_name = f"ModelNet{name}" if dataset_name is None else dataset_name
    pre_transform = T.Compose([T.NormalizeScale(), IncludeMeshFeatures(num_points=num_point_to_sample), AddWalkFeature(k=default_args["k"])])
    transform = T.Compose([EdgeIndexTransform(),]) #AddCentralityNodeFeatures()])
    
    pre_filter_func = pre_filter_f if use_pre_filter else None
    train_dataset_,test_dataset_ = None,None
    if include_train:
        print("Loading train dataset...")
        train_dataset_ = ModelNet(root=dataset_name, name=name, train=True, transform=transform,
                                pre_transform=pre_transform, force_reload=False, pre_filter=pre_filter_func)
        print(f"Train dataset loaded with {len(train_dataset_)} samples")
    if include_test:
        print("Loading test dataset...")
        test_dataset_ = ModelNet(root=dataset_name, name=name, train=False, transform=transform,
                             pre_transform=pre_transform, force_reload=False, pre_filter=pre_filter_func)
        print(f"Test dataset loaded with {len(test_dataset_)} samples")
    return train_dataset_,test_dataset_

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
    dataset_name = f"ModelNet{name}" if dataset_name is None else dataset_name

    pre_transform = T.Compose([
        T.NormalizeScale(),
        T.SamplePoints(num=num_point_to_sample),
        AddWalkFeature(k=k)
    ])
    transform = T.Compose([
        EdgeIndexTransform(),
        #AddCentralityNodeFeatures()
    ])
    
    pre_filter_func = pre_filter_f if use_pre_filter else None
    
    train_dataset_,test_dataset_ = None, None

    if include_train:
        print("Loading train dataset...")
        train_dataset_ = ModelNet(root=dataset_name, name=name, train=True, transform=transform,
                                pre_transform=pre_transform, force_reload=False, pre_filter=pre_filter_func)
        print(f"Train dataset loaded with {len(train_dataset_)} samples")
    
    if include_test:
        print("Loading test dataset...")
        test_dataset_ = ModelNet(root=dataset_name, name=name, train=False, transform=transform,
                             pre_transform=pre_transform, force_reload=False, pre_filter=pre_filter_func)
        print(f"Test dataset loaded with {len(test_dataset_)} samples")
    return train_dataset_,test_dataset_




def load_from_hf(split, sample_points=default_args["n_sample"]):
    """
    Loads the ModelNet10 processed dataset from the Hugging Face repository
    train_dataset = load_from_hf('train')
    val_dataset = load_from_hf('val')
    val_sample = load_from_hf('val_sample')
    """
    repo_id = "talpe/ModelNet10"
    HF_TOKEN = "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"
    api = HfApi(token=HF_TOKEN)
    file_name = f"walks_{split}.pt"  # [walks_train,walks_val,walks_val_sample]
    
    dataset_path = hf_hub_download(repo_id=repo_id, filename=file_name, repo_type="dataset", token=HF_TOKEN)
    loaded_dataset = torch.load(dataset_path)
    loaded_dataset.pre_transform = T.Compose([T.NormalizeScale(), T.SamplePoints(sample_points), AddWalkFeature(k=default_args["k"])])
    # loaded_dataset.transform = T.Compose([T.SamplePoints(sample_points), AddWalkFeature(k=default_args["k"])])
    return loaded_dataset




def create_dataloaders(train_dataset, val_dataset, batch_size=8, num_workers=4):
    """
    Create dataloaders from the datasets which merges the data objects into mini-batches and wrap an iterable around it.
    We use the torch_geometric.loader.DataLoader to create our dataloaders
    :return:
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader


def save_to_hf(split: str, path="ModelNet10"):
    folder_path = f"modelnet10_walks_{split}"
    HF_TOKEN = "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"
    api = HfApi(token=HF_TOKEN)
    api.upload_folder(folder_path=folder_path, repo_id="talpe/ModelNet10", repo_type="dataset", )



def split_data(data, train_size, test_size, val_size):
    n_nodes = data.size[0]
    data_indices = np.range(n_nodes)

    idx_test = np.random.choice(data_indices, test_size, replace=False)
    test_mask = torch_geometric.utils.index_to_mask(idx_test,size=n_nodes)

    test_indices = torch_geometric.utils.mask_select(src=data_indices,dim=0,mask=test_mask)
    rest_indices = torch_geometric.utils.mask_select(src=data_indices,dim=0,mask=~test_mask)

    idx_val = np.random.choice(rest_indices, val_size, replace=False)
    index_to_mask = torch_geometric.utils.index_to_mask(idx_val,size=n_nodes)
    val_indices = torch_geometric.utils.mask_select(src=data_indices,dim=0, mask =index_to_mask)
    train_indices = torch_geometric.utils.mask_select(src=data_indices, mask =~index_to_mask)

    return train_indices, val_indices, test_indices