from dataset import PointCloudDataset
from torch.utils.data import DataLoader
from dataset import WalksDataset
# Load raw point cloud dataset
raw_dataset = PointCloudDataset("/home/cohen-idan/finalproj/Preprocessing/datasets_processed/modelnet40_normal_resampled/train")
raw_dataloader = DataLoader(raw_dataset, batch_size=32, shuffle=True)
for vertices, labels, model_ids in raw_dataloader:
    print("Raw Point Cloud Dataset:")
    print("  Vertices shape:", vertices.shape)  # Expected: (batch_size, 5000, 3)
    print("  Labels shape:", labels.shape)  # Expected: (batch_size,)
    print("  Model IDs:", model_ids)  # List of model names
    break  # Only print first batch


walks_dataset = WalksDataset("/home/cohen-idan/finalproj/Preprocessing/pre_created_walks")
walks_dataloader = DataLoader(walks_dataset, batch_size=4, shuffle=True)
for walks, labels, model_ids in walks_dataloader:
    print("Walks Dataset:")
    print("  Walks shape:", walks.shape)  # Expected: (batch_size, num_walks, seq_len, 3)
    print("  Labels shape:", labels.shape)  # Expected: (batch_size,)
    print("  Model IDs:", model_ids)  # List of model names
    break  # Only print first batch