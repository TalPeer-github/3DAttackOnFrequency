import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import PointCloudDataset
import json
import argparse
from scipy.spatial import KDTree
import random

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

def generate_random_walks(vertices, num_walks, seq_len, k_neighbors):
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

        save_traj_as_npz(file_name[0], walks, label, save_path)

    print(f"Walks saved successfully to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random walks for ModelNet40")
    parser.add_argument("--dataset", type=str, choices=["train", "test"], required=True, help="Dataset to process (train/test)")
    args = parser.parse_args()

    dataset_path = params["train_path"] if args.dataset == "train" else params["test_path"]
    save_path = os.path.join(params["save_path"], args.dataset)

    generate_walks(dataset_path, save_path)
