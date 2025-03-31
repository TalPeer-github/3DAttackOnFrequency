import os
import sys
import numpy as np
import torch

def inspect_npz(file_path):
    """
    Inspect the contents of an NPZ file and print detailed information
    about each array and stored object.
    """
    print(f"\nInspecting file: {file_path}")
    print("=" * 50)
    
    # Load the NPZ file
    data = np.load(file_path, allow_pickle=True)
    
    # Print available keys
    print("\nAvailable arrays/objects:")
    print("-" * 30)
    for key in data.files:
        print(f"Key: {key}")
    
    # Detailed inspection of each array/object
    print("\nDetailed Information:")
    print("-" * 30)
    
    for key in data.files:
        array = data[key]
        print(f"\nKey: {key}")
        print(f"Type: {type(array)}")
        print(f"Shape: {array.shape if hasattr(array, 'shape') else 'N/A'}")
        
        # Special handling for different types of data
        if key == 'info' and array.dtype == np.dtype('O'):
            info_dict = array.item()
            print("\nInfo dictionary contents:")
            for k, v in info_dict.items():
                print(f"  {k}: {v}")
        
        elif key in ['walks', 'original_walks']:
            print(f"Data type: {array.dtype}")
            print(f"Min value: {array.min()}")
            print(f"Max value: {array.max()}")
            print(f"Mean value: {array.mean()}")
            if len(array.shape) == 3:
                print(f"Number of walks: {array.shape[0]}")
                print(f"Points per walk: {array.shape[1]}")
                print(f"Dimensions: {array.shape[2]}")
        
        elif key == 'label':
            print(f"Label value: {array.item()}")
        
        else:
            try:
                print(f"Content: {array}")
            except:
                print("Content too large to display")

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_npz.py <path_to_npz_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)
    
    inspect_npz(file_path)

if __name__ == "__main__":
    main() 