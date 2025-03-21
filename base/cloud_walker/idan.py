import numpy as np
from dataset import tf_point_cloud_dataset
from easydict import EasyDict

# Define dataset parameters
params = EasyDict({
    'batch_size': 8,        # Number of walks per batch
    'seq_len': 32,          # Walk length
    'walk_alg': 'local_jumps',  
    'net_input': ['xyz']    # Features included in the walks
})

# Load the dataset (This will generate the walks)
dataset, num_samples = tf_point_cloud_dataset(params, "datasets_processed-tmp/modelnet40_normal_resampled/*")

# Extract a batch of walks
for batch in dataset.take(1):  # Get the first batch
    filenames, features, labels = batch  # Extract batch data
    
    print(f"Batch Shape: {features.shape}")  # Expected: (batch_size, seq_len, feature_dim)
    print(f"Example Walk Data (First Walk):\n{features.numpy()[0]}")
    
    # Save batch for the imitating network
    np.savez("pre_created_walks/batch_walks.npz", features=features.numpy(), labels=labels.numpy())

    break  # Stop after processing one batch

import numpy as np
import utils

# Load the saved walks
data = np.load("pre_created_walks/batch_walks.npz")
walks = data['features']

# Load a sample model for visualization
sample_model = np.load("datasets_processed-tmp/modelnet40_normal_resampled/test__5000__airplane__airplane_0001.npz")
vertices = sample_model['vertices']

# Visualize the first walk
utils.visualize_model_walk(vertices, [walks[0]], seed=42)
