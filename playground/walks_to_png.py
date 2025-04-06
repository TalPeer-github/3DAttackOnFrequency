import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for Linux
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set the correct file paths
walks_path = "/home/cohen-idan/finalproj/Preprocessing/pre_created_walks/test/car_0204/car_0204_traj.npz"
original_path = "/home/cohen-idan/finalproj/Preprocessing/datasets_processed/modelnet40_normal_resampled/test/test__5000__car__car_0204.npz"

# ✅ Load the generated walks file
data = np.load(walks_path)
print(data)
walks = data["model_features"]
label = data["label"]

# ✅ Load the original object (point cloud)
original_data = np.load(original_path)
vertices = original_data["vertices"]  # Shape: (5000, 3)

# ✅ Ensure the walk shape is correct
if walks.shape[0] == 1:
    walks = walks.squeeze(0)  # Expected: (32, 800, 3)

# Debugging: Print shape after squeeze
print(f"Walks Shape After Squeeze: {walks.shape}")  # Expected: (32, 800, 3)

# ✅ Create a directory for saving all walks
save_dir = "/home/cohen-idan/finalproj/walks_vis"
os.makedirs(save_dir, exist_ok=True)

# ✅ Plot and save all 32 walks
for i in range(walks.shape[0]):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original object as a scatter plot (light gray)
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c="gray", s=1, alpha=0.2, label="Object")

    # Plot the current walk
    ax.plot(walks[i, :, 0], walks[i, :, 1], walks[i, :, 2], marker="o", linestyle="-", markersize=2, alpha=0.8, label=f"Walk {i+1}")

    # Labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Save each walk separately
    plot_path = os.path.join(save_dir, f"walk_{i+1}.png")
    plt.savefig(plot_path)
    plt.close(fig)  # Close figure to save memory

    print(f"Saved: {plot_path}")

print(f"All 32 walks saved in {save_dir}")
