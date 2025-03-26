import os
import glob
import json
import argparse
from types import SimpleNamespace

from dataset import WalksDataset, PointCloudDataset
from attack_single_pc import attack_single_pc


def attack_all_walks(cfg):
    walk_dataset = WalksDataset(cfg.walk_npz_root)
    pc_dataset = PointCloudDataset(cfg.original_pc_root)
    output_root = cfg.output_dir

    # Load categories from walks_creating.json
    walks_config_path = os.path.join("configs", "walks_creating.json")
    if os.path.exists(walks_config_path):
        with open(walks_config_path, "r") as f:
            walks_config = json.load(f)
        selected_categories = set(walks_config.get("categories", []))
    else:
        raise ValueError("walks_creating.json not found or missing 'categories' key")

    # Collect model_ids from point cloud folder, filter by category prefix
    pc_files = glob.glob(os.path.join(cfg.original_pc_root, "*.npz"))
    model_ids = []
    for path in pc_files:
        basename = os.path.basename(path).replace(".npz", "")
        model_id = basename.split("__")[-1]  # e.g., airplane_0631
        category_prefix = model_id.split("_")[0]
        if category_prefix in selected_categories:
            model_ids.append(model_id)

    print(f"[INFO] Found {len(model_ids)} model IDs to attack from {len(selected_categories)} categories.")

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
