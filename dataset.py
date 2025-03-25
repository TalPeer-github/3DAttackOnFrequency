import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import KDTree
import random


# ----------------------- POINT CLOUD DATASET -----------------------

class PointCloudDataset(Dataset):
    """
    Dataset for loading raw point clouds (5000 points of x,y,z) from ModelNet40.
    Includes normalization and optional augmentations.
    """

    def __init__(self, dataset_path, augment=False):
        """
        Args:
            dataset_path (str): Path to the directory containing .npz raw point cloud files.
            augment (bool): Whether to apply data augmentation.
        """
        self.files = sorted(glob.glob(os.path.join(dataset_path, "*.npz")))
        self.augment = augment  # Enable or disable augmentation
        self.id_to_index = {}  # model_id → index
        for i, file in enumerate(self.files):
            filename = os.path.basename(file).replace(".npz", "")  # e.g., train__5000__airplane__airplane_0001
            model_id = filename.split("__")[-1]  # Extracts airplane_0001
            self.id_to_index[model_id] = i

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Loads a raw point cloud file.

        Args:
            idx (int): Index of the file to load.

        Returns:
            tuple: (vertices, label, file_id)
                - vertices: Tensor of shape (5000, 3) containing x, y, z coordinates.
                - label: Tensor containing the object class label.
                - file_id: String with the model identifier.
        """
        file_path = self.files[idx]
        data = np.load(file_path, allow_pickle=True)

        vertices = torch.tensor(data["vertices"], dtype=torch.float32)  # (5000, 3)
        label = torch.tensor(data["label"], dtype=torch.long)  # Integer label
        model_id = os.path.basename(file_path).replace(".npz", "").split("__")[-1]  # airplane_0001

        # Normalize point cloud
        vertices = self.normalize_point_cloud(vertices)

        # Apply augmentation if enabled
        if self.augment:
            vertices = self.apply_augmentation(vertices)

        return vertices, label, model_id

    def get_by_model_id(self, model_id):
        idx = self.id_to_index.get(model_id)
        if idx is None:
            raise ValueError(f"Model ID {model_id} not found in dataset.")
        return self[idx]

    def normalize_point_cloud(self, vertices):
        """Normalize point cloud to unit sphere."""
        mean = torch.mean(vertices, dim=0)
        vertices -= mean  # Center at origin
        max_norm = torch.max(torch.norm(vertices, dim=1))
        vertices /= max_norm  # Scale to unit sphere
        return vertices

    def apply_augmentation(self, vertices):
        """Applies a set of augmentations."""
        vertices = self.rotate_point_cloud(vertices)
        vertices = self.jitter_point_cloud(vertices)
        vertices = self.random_scale_point_cloud(vertices)
        vertices = self.shift_point_cloud(vertices)
        return vertices

    def rotate_point_cloud(self, vertices):
        """Rotate point cloud around the Y-axis."""
        theta = np.random.uniform(0, 2 * np.pi)  # Random rotation angle
        rot_matrix = torch.tensor([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ], dtype=torch.float32)
        return torch.mm(vertices, rot_matrix)

    def jitter_point_cloud(self, vertices, sigma=0.01, clip=0.05):
        """Apply random jitter to the point cloud."""
        noise = torch.clamp(sigma * torch.randn(vertices.shape), -clip, clip)
        return vertices + noise

    def random_scale_point_cloud(self, vertices, scale_low=0.8, scale_high=1.25):
        """Apply random scaling to the point cloud."""
        scale = np.random.uniform(scale_low, scale_high)
        return vertices * scale

    def shift_point_cloud(self, vertices, shift_range=0.1):
        """Apply random translation to the point cloud."""
        shift = torch.tensor(np.random.uniform(-shift_range, shift_range, 3), dtype=torch.float32)
        return vertices + shift


# ----------------------- WALKS DATASET -----------------------

class WalksDataset(Dataset):
    """
    Dataset for loading precomputed random walks from .npz files.
    """

    def __init__(self, dataset_path):
        """
        Args:
            dataset_path (str): Path to the directory containing precomputed walks.
        """
        # Find all subdirectories inside dataset_path
        self.folders = sorted(glob.glob(os.path.join(dataset_path, "*")))
        
        # Collect all _traj.npz files inside those folders
        self.files = [
            os.path.join(folder, os.path.basename(folder) + "_traj.npz")
            for folder in self.folders
            if os.path.exists(os.path.join(folder, os.path.basename(folder) + "_traj.npz"))
        ]
        self.id_to_index = {}
        for idx, path in enumerate(self.files):
            filename = os.path.basename(path).replace("_traj.npz", "")  # e.g., test__5000__airplane__airplane_0001
            model_id = filename.split("__")[-1]  # Extract airplane_0001
            self.id_to_index[model_id] = idx

        #print(f"Keys: {list(self.id_to_index.keys())}")
        if not self.files:
            raise ValueError(f"No _traj.npz files found in {dataset_path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Loads a precomputed walk file.

        Args:
            idx (int): Index of the file to load.

        Returns:
            tuple: (walks, label, file_id)
                - walks: Tensor of shape (num_walks, seq_len, 3)
                - label: Tensor representing the class label.
                - file_id: String with the model identifier.
        """
        file_path = self.files[idx]
        data = np.load(file_path, allow_pickle=True)
        
        model_features = torch.tensor(data["model_features"], dtype=torch.float32)  # (num_walks, seq_len, 3)
        label = torch.tensor(data["label"], dtype=torch.long)  # Integer label
        model_id = os.path.basename(file_path).replace("_traj.npz", "").split("__")[-1]  # airplane_0001

        return model_features, label, model_id
    
    def get_by_model_id(self, model_id):
        print(f"Model ID: {model_id}")
        if model_id not in list(self.id_to_index.keys()):
            raise ValueError(f"[ERROR] model_id {model_id} not found in WalksDataset.")
        return self[self.id_to_index[model_id]]