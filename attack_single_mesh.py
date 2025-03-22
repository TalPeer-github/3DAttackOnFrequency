import os
import time
import json
import torch
import numpy as np
from types import SimpleNamespace
from tqdm import tqdm
import torch.nn.functional as F

from dataset import WalksDataset
from proxy_network import RnnWalkNet


def define_network_and_its_params(config):
    """
    Loads model parameters and the trained RNN model (CloudWalker).
    """
    with open(os.path.join(config["trained_model"], 'params.txt')) as f:
        params_dict = json.load(f)
    params = SimpleNamespace(**params_dict)

    # Override or ensure runtime consistency
    params.batch_size = 1
    params.seq_len = config["walk_len"]
    params.n_walks_per_model = 32
    params.use_norm_layer = config.get("use_norm_layer", "BatchNorm")
    params.layer_sizes = None

    model = RnnWalkNet(params, config["num_classes"], net_input_dim=3)
    checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return params, model


def load_walk_npz_by_id(model_id, dataset_root):
    """
    Find the walk .npz file for the given model ID (e.g., airplane_0123)
    """
    model_dir = os.path.join(dataset_root, model_id)
    walk_path = os.path.join(model_dir, model_id + "_traj.npz")
    if not os.path.exists(walk_path):
        raise FileNotFoundError(f"Walk file not found: {walk_path}")
    return walk_path


def attack_single_mesh(config, model_id, output_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params, dnn_model = define_network_and_its_params(config)
    dnn_model.to(device)

    # Load walk from .npz
    walk_path = load_walk_npz_by_id(model_id, config["walk_npz_root"])
    data = np.load(walk_path, allow_pickle=True)
    walks = torch.tensor(data["model_features"], dtype=torch.float32)  # (N, T, 3)
    label = int(data["label"])

    walks = walks.to(device)
    walks.requires_grad = True

    # One-hot label
    one_hot = F.one_hot(torch.tensor(label), num_classes=config["num_classes"]).float().to(device)

    # Attack hyperparams
    w = config.get("attacking_weight", 1.0)
    step_size = config.get("step_size", 0.01)
    max_iter = config.get("max_iter", 50)
    max_label_diff = config.get("max_label_diff", 0.01)

    num_wrong = 0
    for step in range(max_iter):
        dnn_model.zero_grad()
        logits = dnn_model(walks)  # shape (N, C)
        probs = F.softmax(logits, dim=1)
        avg_pred = probs.mean(dim=0)

        loss = -w * F.kl_div(avg_pred.log(), one_hot, reduction="batchmean")
        loss.backward()
        grad = walks.grad.detach()

        # Apply gradient step
        walks = walks + step_size * grad.sign()
        walks = walks.detach().clone().requires_grad_(True)

        # Evaluate after step
        new_logits = dnn_model(walks)
        new_probs = F.softmax(new_logits, dim=1).mean(dim=0)
        pred_class = torch.argmax(new_probs).item()
        confidence = new_probs[label].item()

        print(f"[{step:03d}] Loss: {loss.item():.4f} | True Conf: {confidence:.4f} | Pred: {pred_class}")

        if pred_class != label:
            num_wrong += 1
        else:
            num_wrong = 0

        if num_wrong >= 10:
            print("Early stopping: model consistently fooled.")
            break

    # Save attacked walk
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{model_id}_attacked.npz")
        np.savez(out_path, model_features=walks.cpu().numpy(), label=label, model_id=model_id)
        print(f"Saved attacked walk to: {out_path}")

    return walks.detach().cpu()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="attack_config.json")
    parser.add_argument("--id", type=str, required=True, help="Model ID, e.g., airplane_0123")
    parser.add_argument("--output_dir", type=str, default="attacks")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    attack_single_mesh(config, args.id, output_dir=args.output_dir)
