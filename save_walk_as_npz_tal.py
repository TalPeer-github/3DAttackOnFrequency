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


def generate_random_walks_prev(vertices, num_walks, seq_len, k_neighbors):
    """ Generate multiple random walks using k-NN """
    tree = KDTree(vertices)
    walks = []

    for _ in range(num_walks):
        start_idx = np.random.randint(len(vertices))
        seq = [start_idx]

        for _ in range(seq_len - 1):
            neighbors = tree.query(vertices[start_idx], k=k_neighbors)[1]
            next_idx = random.choice(neighbors)
            seq.append(next_idx)
            start_idx = next_idx

        walks.append(vertices[seq])

    return np.array(walks)  # Shape: (num_walks, seq_len, 3)


from scipy.spatial import KDTree
import numpy as np
import torch

def generate_random_walks(vertices, num_walks=1, seq_len=10, k_neighbors=8):
    """
    Generate random walks over the point cloud `vertices`.
    
    Each walk starts at a random vertex and proceeds by selecting from the k-nearest
    unvisited neighbors. If no unvisited neighbors are available, a jump to a new unvisited
    vertex is made.
    
    Args:
        vertices (np.ndarray): Array of shape (N, 3), raw point cloud coordinates.
        num_walks (int): Number of walks to generate.
        seq_len (int): Number of steps in each walk.
        k_neighbors (int): Number of neighbors to sample from at each step.
    
    Returns:
        torch.Tensor: Random walks of shape (num_walks, seq_len+1, 3)
    """
    kd_tree = KDTree(vertices)
    n_vertices = len(vertices)

    total_walks = []
    optional_starts = list(range(n_vertices))

    for _ in range(num_walks):
        start_point = np.random.choice(optional_starts)
        optional_starts.remove(start_point)

        walk = np.full((seq_len,), -1, dtype=np.int32)
        walk[0] = start_point

        jumps = np.zeros((seq_len + 1,), dtype=bool)

        visited = np.zeros(n_vertices + 1, dtype=bool)
        visited[-1] = True
        visited[start_point] = True

        for walk_step in range(1, seq_len + 1):
            current_point = vertices[walk[walk_step - 1]]
            distances, neighbors = kd_tree.query(current_point, k=k_neighbors * 2)
            neighbors = np.atleast_1d(neighbors)

            neighbors_to_consider = [n for n in neighbors if not visited[n]]
            if neighbors_to_consider:
                chosen = np.random.choice(neighbors_to_consider[:k_neighbors])
                walk[walk_step] = chosen
                jumps[walk_step] = False
            else:
                new_point = np.random.choice([i for i in range(n_vertices) if not visited[i]])
                walk[walk_step] = new_point
                jumps[walk_step] = True
                visited = np.zeros(n_vertices + 1, dtype=bool)
                visited[-1] = True

            visited[walk[walk_step]] = True

        walk_coords = vertices[walk]  # shape: (seq_len+1, 3)
        total_walks.append(torch.tensor(walk_coords, dtype=torch.float32))

    return torch.stack(total_walks)  # shape: (num_walks, seq_len+1, 3)



def generate_walks(dataset_path, save_path):
    """ Loads data, generates walks, and saves them """
    dataset = PointCloudDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"Processing dataset: {dataset_path}")

    for vertices, label, file_name in dataloader:
        vertices = vertices.squeeze(0).numpy()
        label = label.item()

        walks = generate_random_walks(
            vertices,
            num_walks=params["num_walks_per_sample"],
            seq_len=params["seq_len"],
            k_neighbors=params["k_neighbors"]
        )

        print(f"File name: {file_name}")
        save_traj_as_npz(file_name[0], walks, label, save_path)
        break
    
    print(f"Walks saved successfully to {save_path}")


if __name__ == "__main__":
    print(f"[{datetime.now()}] Starting process save_walk_as_npz...")
    parser = argparse.ArgumentParser(description="Generate random walks for ModelNet40")
    parser.add_argument("--dataset", type=str, choices=["train", "test"], required=True,
                        help="Dataset to process (train/test)")
    args = parser.parse_args()

    dataset_path = params["train_path"] if args.dataset == "train" else params["test_path"]
    save_path = os.path.join(params["save_path"], args.dataset)

    generate_walks(dataset_path, save_path)
    print(f"[{datetime.now()}] Ending process save_walk_as_npz...")
