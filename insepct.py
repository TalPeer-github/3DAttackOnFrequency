import numpy as np
import argparse
import os

def inspect_npz(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    data = np.load(file_path)
    print(f"Loaded: {file_path}")
    print("Contained arrays:")

    for key in data.files:
        arr = data[key]
        print(f"  '{key}': shape {arr.shape}, dtype {arr.dtype}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect .npz file contents.")
    parser.add_argument("path", help="Path to the .npz file")
    args = parser.parse_args()
    inspect_npz(args.path)
