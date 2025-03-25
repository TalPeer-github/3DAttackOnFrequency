import os
import json
import argparse
import re
import numpy as np
from attack_single_pc import attack_single_pc
from dataset import PointCloudDataset, WalksDataset
from types import SimpleNamespace

# CLASS LABELS from ModelNet40
modelnet40_labels = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
    'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
    'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
]
modelnet40_labels.sort()

import glob
import os

def attack_all_walks(cfg):
    from dataset import WalksDataset, PointCloudDataset  # Ensure this import matches your structure

    walk_dataset = WalksDataset(cfg.walk_npz_root)
    pc_dataset = PointCloudDataset(cfg.original_pc_root)
    output_root = cfg.output_dir

    pc_files = glob.glob(os.path.join(cfg.original_pc_root, "*.npz"))
    model_ids = [
        os.path.basename(path).replace(".npz", "").split("__")[-1]
        for path in pc_files
    ]

    print(f"[INFO] Found {len(model_ids)} model IDs to attack.")

    for model_id in model_ids:
        try:
            attack_single_pc(
                cfg=cfg,
                model_id=model_id,
                walk_dataset=walk_dataset,
                pc_dataset=pc_dataset,
                output_dir=output_root,
            )
        except Exception as e:
            print(f"[ERROR] Failed on {model_id}: {e}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/attack_config.json")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = json.load(f)
    cfg = SimpleNamespace(**cfg_dict)

    attack_all_walks(cfg)


if __name__ == "__main__":
    main()
