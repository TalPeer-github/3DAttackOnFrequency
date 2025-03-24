import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
import torch.nn.functional as F

from dataset import WalksDataset
from proxy_network import RnnWalkNet


def define_network_and_its_params(cfg):
    """
    Loads CloudWalker RNN model from config checkpoint and initializes it.
    """
    model = RnnWalkNet(cfg, cfg.num_classes, net_input_dim=3)
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


import os

def load_walk_npz_by_id(model_id, dataset_root):
    """
    Loads a single walk .npz file from disk given a model ID (e.g., airplane_0631).
    It converts it into the full path: 
    dataset_root/test__5000__airplane__airplane_0631/test__5000__airplane__airplane_0631_traj.npz
    """
    #change to match file names
    name, idx = model_id.split("_")
    folder_name = f"test__5000__{name}__{model_id}"
    file_name = f"{folder_name}_traj.npz"
    full_path = os.path.join(dataset_root, folder_name, file_name)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"[ERROR] Walk file not found: {full_path}")

    return full_path



def attack_single_pc(cfg, model_id, output_dir=None):
    print(f"[INFO] Attacking: {model_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = define_network_and_its_params(cfg).to(device)
    model.train()
    # Load walk input
    walk_path = load_walk_npz_by_id(model_id, cfg.walk_npz_root)
    data = np.load(walk_path, allow_pickle=True)
    walks = torch.tensor(data["model_features"], dtype=torch.float32).to(device)  # (N, T, 3)
    label = int(data["label"])
    walks.requires_grad = True

    # Attack params
    w = cfg.attacking_weight
    step_size = cfg.step_size
    max_iter = cfg.max_iter
    max_label_diff = cfg.max_label_diff

    # One-hot true label
    one_hot = F.one_hot(torch.tensor(label), num_classes=cfg.num_classes).float().to(device)
    num_wrong = 0

    for step in range(max_iter):
        model.zero_grad()
        logits = model(walks)  # (N, C)
        probs = F.softmax(logits, dim=1)
        avg_pred = probs.mean(dim=0)  # (C,)

        loss = -w * F.kl_div(avg_pred.log(), one_hot, reduction="batchmean")
        loss.backward()
        grad = walks.grad.detach()

        # Gradient step
        walks = walks + step_size * grad.sign()
        walks = walks.detach().clone().requires_grad_(True)

        # Re-evaluate
        new_logits = model(walks)
        new_probs = F.softmax(new_logits, dim=1).mean(dim=0)
        pred_class = torch.argmax(new_probs).item()
        confidence = new_probs[label].item()

        print(f"[{step:03d}] Loss: {loss.item():.4f} | True Conf: {confidence:.4f} | Pred: {pred_class}")

        if pred_class != label:
            num_wrong += 1
        else:
            num_wrong = 0

        if num_wrong >= 10:
            print("[✔] Early stopping: model consistently fooled.")
            break

    # Save result
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{model_id}_attacked.npz")
        np.savez(out_path, model_features=walks.detach().cpu().numpy(), label=label, model_id=model_id)
        print(f"[✓] Saved attacked walk to: {out_path}")


    return walks.detach().cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/attack_config.json")
    parser.add_argument("--id", type=str, required=True, help="Model ID, e.g., airplane_0123")
    parser.add_argument("--output_dir", type=str, default="attacks")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = json.load(f)
    cfg = SimpleNamespace(**cfg_dict)

    attack_single_pc(cfg, args.id, output_dir=args.output_dir)
