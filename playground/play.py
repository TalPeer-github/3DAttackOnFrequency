import numpy as np

# Load the generated walk file
data = np.load("/home/cohen-idan/finalproj/Preprocessing/pre_created_walks/train__5000__airplane__airplane_0297.npz/train__5000__airplane__airplane_0297.npz_traj.npz")
# Extract data
walks = data["model_features"]  # Shape: (num_walks, walk_length, 3)
label = data["label"]  # Integer class label (e.g., 0 for airplane)
model_id = data["model_id"]  # Original file name

print("Walks Shape:", walks.shape)  # Example: (48, 800, 3)
print("Label:", label)  # Example: 0 (airplane)
print("Model ID:", model_id)  # Example: "train_5000__airplane__airplane_0001.npz"
