import os
import json
import argparse
import re
import numpy as np
from attack_single_pc import attack_single_pc


# CLASS LABELS from ModelNet40
modelnet40_labels = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
    'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
    'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
]
modelnet40_labels.sort()

def attack_all_walks(config):
    np.random.seed(0)

    dataset_root = config["walk_npz_root"]
    output_root = config.get("output_dir", "attacks")

    for label_idx, class_name in enumerate(modelnet40_labels):
        config["source_label"] = label_idx
        class_folder = os.path.join(dataset_root)
        model_dirs = [d for d in os.listdir(class_folder) if d.startswith(class_name)]

        for model_id in model_dirs:
            npz_path = os.path.join(class_folder, model_id, f"{model_id}_traj.npz")
            attacked_path = os.path.join(output_root, f"{model_id}_attacked.npz")

            if os.path.exists(attacked_path):
                print(f"Skipping {model_id}: already attacked")
                continue

            print(f"\n[Attacking] Class: {class_name} (label {label_idx}) | Model ID: {model_id}")
            try:
                attack_single_pc(config=config, model_id=model_id, output_dir=output_root)
            except Exception as e:
                print(f"Failed to attack {model_id}: {e}")
                continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/attack_config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    attack_all_walks(config)


if __name__ == "__main__":
    main()
