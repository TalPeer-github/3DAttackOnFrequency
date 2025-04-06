import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import PointCloudDataset
import json
import argparse
from scipy.spatial import KDTree
import random
from datetime import datetime

# Load configuration
with open("configs/walks_creating.json", "r") as f:
    params = json.load(f)

def save_traj_as_npz(file_name, model_features, label, save_path):
    """ Saves the generated walk as .npz """
    save_dir = os.path.join(save_path, file_name)
    os.makedirs(save_dir, exist_ok=True)

    np.savez(
        os.path.join(save_dir, f"{file_name}_traj.npz"),
        model_features=np.asarray(model_features),
        label=label,
        model_id=file_name
    )

import numpy as np
import torch
from scipy.spatial import KDTree

def generate_random_walks_tensor(pc_tensor, num_walks=32, seq_len=256, k_neighbors=8):
    """
    Pure PyTorch differentiable random walk generator.
    
    Args:
        pc_tensor: (N, 3) torch tensor with gradients enabled
        Returns: (num_walks, seq_len) LongTensor of indices into pc_tensor
    """
    N = pc_tensor.shape[0]
    dists = torch.cdist(pc_tensor, pc_tensor)  # (N, N)
    knn_indices = torch.topk(dists, k=k_neighbors + 1, largest=False).indices[:, 1:]  # skip self

    walks = torch.zeros((num_walks, seq_len), dtype=torch.long, device=pc_tensor.device)
    
    for w in range(num_walks):
        start = torch.randint(0, N, (1,), device=pc_tensor.device).item()
        walk = [start]
        visited = {start}

        for _ in range(seq_len - 1):
            neighbors = knn_indices[walk[-1]]
            unvisited = [n.item() for n in neighbors if n.item() not in visited]

            if unvisited:
                next_pt = unvisited[torch.randint(0, len(unvisited), (1,)).item()]
            else:
                next_pt = torch.randint(0, N, (1,), device=pc_tensor.device).item()
                visited = set()

            walk.append(next_pt)
            visited.add(next_pt)

        walks[w] = torch.tensor(walk, dtype=torch.long, device=pc_tensor.device)

    return walks  # shape: (num_walks, seq_len)

def generate_random_walks(vertices, num_walks=32, seq_len=256, k_neighbors=8):
    """
    Generate random walks over the point cloud.

    Args:
        vertices (np.ndarray): Array of shape (N, 3) representing the point cloud.
        num_walks (int): Number of random walks to generate.
        seq_len (int): Length of each walk (number of steps).
        k_neighbors (int): Number of nearest neighbors to consider during walk.

    Returns:
        torch.Tensor: Random walks of shape (num_walks, seq_len, 3)
    """
    kd_tree = KDTree(vertices)
    n_vertices = len(vertices)

    total_walks = []
    for _ in range(num_walks):
        # Step 1: Start at a random point
        start_point = np.random.randint(0, n_vertices)
        walk_indices = [start_point]
        visited = set(walk_indices)

        for _ in range(seq_len - 1):
            current_point = vertices[walk_indices[-1]]
            _, neighbors = kd_tree.query(current_point, k=k_neighbors * 2)

            # Filter out visited
            unvisited = [idx for idx in neighbors if idx not in visited]

            if unvisited:
                next_point = np.random.choice(unvisited)
            else:
                next_point = np.random.randint(0, n_vertices)
                visited = set()  # Reset visited if stuck

            walk_indices.append(next_point)
            visited.add(next_point)

        walk_coords = vertices[walk_indices]  # shape (seq_len, 3)
        total_walks.append(torch.tensor(walk_coords, dtype=torch.float32))

    total_walks = torch.stack(total_walks)  # shape (num_walks, seq_len, 3) - 3 for x,y,z coordinates. 
    return total_walks


def generate_walks(dataset_path, save_path):
    """ Loads data, generates walks, and saves them """
    dataset = PointCloudDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"Processing dataset: {dataset_path}")

    for vertices, label, file_name in dataloader:
        vertices = vertices.squeeze(0).numpy()
        label = label.item()  # This label is already remapped by the PointCloudDataset

        walks = generate_random_walks(
            vertices,
            num_walks=params["num_walks_per_sample"],
            seq_len=params["seq_len"],
            k_neighbors=params["k_neighbors"]
        )
        
        save_traj_as_npz(file_name[0], walks, label, save_path)
    
    print(f"Walks saved successfully to {save_path}")

if __name__ == "__main__":
    print(f"[{datetime.now()}] Starting process save_walk_as_npz...")
    parser = argparse.ArgumentParser(description="Generate random walks for ModelNet40")
    parser.add_argument("--dataset", type=str, choices=["train", "test"], required=True, help="Dataset to process (train/test)")
    args = parser.parse_args()

    dataset_path = params["train_path"] if args.dataset == "train" else params["test_path"]
    save_path = os.path.join(params["save_path"], args.dataset)
    generate_walks(dataset_path, save_path)
    print(f"[{datetime.now()}] Ending process save_walk_as_npz...")