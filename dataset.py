import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import json

# Load category subset from config
walks_config_path = os.path.join("configs", "walks_creating.json")
if os.path.exists(walks_config_path):
    with open(walks_config_path, "r") as f:
        _walks_config = json.load(f)
    categories = _walks_config.get("categories", None)
else:
    categories = None

# Category to new label mapping (0...N)
category_to_idx = {category: idx for idx, category in enumerate(categories)} if categories else None
if category_to_idx:
    print(f"[INFO] Mapping categories to new indices: {category_to_idx}")


class PointCloudDataset(Dataset):
    def __init__(self, dataset_path, augment=False):
        all_files = sorted(glob.glob(os.path.join(dataset_path, "*.npz")))
        if categories:
            self.files = [
                f for f in all_files
                if any(os.path.basename(f).split("__")[-1].startswith(cat + "_") for cat in categories)
            ]
        else:
            self.files = all_files

        self.augment = augment
        self.id_to_index = {}
        for i, file in enumerate(self.files):
            model_id = os.path.basename(file).replace(".npz", "").split("__")[-1]
            self.id_to_index[model_id] = i

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path, allow_pickle=True)
        vertices = torch.tensor(data["vertices"], dtype=torch.float32)
        model_id = os.path.basename(file_path).replace(".npz", "").split("__")[-1]

        vertices = self.normalize_point_cloud(vertices)
        if self.augment:
            vertices = self.apply_augmentation(vertices)

        if categories:
            category = model_id.split("_")[0]
            if category not in category_to_idx:
                raise ValueError(f"[ERROR] Unknown category '{category}' in {model_id}")
            label = torch.tensor(category_to_idx[category], dtype=torch.long)
        else:
            label = torch.tensor(data["label"].item(), dtype=torch.long)

        return vertices, label, model_id

    def get_by_model_id(self, model_id):
        idx = self.id_to_index.get(model_id)
        if idx is None:
            raise ValueError(f"Model ID {model_id} not found.")
        return self[idx]

    def normalize_point_cloud(self, vertices):
        mean = torch.mean(vertices, dim=0)
        vertices -= mean
        max_norm = torch.max(torch.norm(vertices, dim=1))
        vertices /= max_norm
        return vertices

    def apply_augmentation(self, vertices):
        vertices = self.rotate_point_cloud(vertices)
        vertices = self.jitter_point_cloud(vertices)
        vertices = self.random_scale_point_cloud(vertices)
        vertices = self.shift_point_cloud(vertices)
        return vertices

    def rotate_point_cloud(self, vertices):
        theta = np.random.uniform(0, 2 * np.pi)
        rot_matrix = torch.tensor([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ], dtype=torch.float32)
        return torch.mm(vertices, rot_matrix)

    def jitter_point_cloud(self, vertices, sigma=0.01, clip=0.05):
        noise = torch.clamp(sigma * torch.randn(vertices.shape), -clip, clip)
        return vertices + noise

    def random_scale_point_cloud(self, vertices, scale_low=0.8, scale_high=1.25):
        scale = np.random.uniform(scale_low, scale_high)
        return vertices * scale

    def shift_point_cloud(self, vertices, shift_range=0.1):
        shift = torch.tensor(np.random.uniform(-shift_range, shift_range, 3), dtype=torch.float32)
        return vertices + shift


class WalksDataset(Dataset):
    def __init__(self, dataset_path):
        self.folders = sorted(glob.glob(os.path.join(dataset_path, "*")))
        self.files = []
        for folder in self.folders:
            category = os.path.basename(folder).split("_")[0]
            if categories and category not in category_to_idx:
                #print(f"[INFO] Skipping category '{category}' not in selected categories.")
                continue
            traj_file = os.path.join(folder, os.path.basename(folder) + "_traj.npz")
            if os.path.exists(traj_file):
                self.files.append(traj_file)

        if not self.files:
            raise ValueError("[ERROR] No valid _traj.npz files after filtering by selected categories.")

        self.id_to_index = {}
        for idx, path in enumerate(self.files):
            model_id = os.path.basename(path).replace("_traj.npz", "").split("__")[-1]
            self.id_to_index[model_id] = idx

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.load(file_path, allow_pickle=True)
        walks = torch.tensor(data["model_features"], dtype=torch.float32)
        model_id = os.path.basename(file_path).replace("_traj.npz", "").split("__")[-1]

        if categories:
            category = model_id.split("_")[0]
            if category not in category_to_idx:
                raise ValueError(f"[ERROR] Unknown category '{category}' in {model_id}")
            label = torch.tensor(category_to_idx[category], dtype=torch.long)
        else:
            label = torch.tensor(data["label"].item(), dtype=torch.long)

        return walks, label, model_id

    def get_by_model_id(self, model_id):
        if model_id not in self.id_to_index:
            raise ValueError(f"[ERROR] model_id {model_id} not found in WalksDataset.")
        return self[self.id_to_index[model_id]]