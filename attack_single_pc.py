import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
import torch.nn.functional as F
from proxy_network import RnnWalkNet
from dataset import PointCloudDataset, WalksDataset

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


# tal function
def attack_single_pc(cfg, model_id, walk_dataset, pc_dataset, output_dir=None):
    print(f"[INFO] Attacking: {model_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = define_network_and_its_params(cfg).to(device)
    model.train()

    # Load walks and point cloud
    walks, label_tensor, walk_id = walk_dataset.get_by_model_id(model_id)
    vertices, _, pc_id = pc_dataset.get_by_model_id(model_id)
    ###
    assert walk_id == pc_id == model_id, f"Mismatch in IDs: {walk_id}, {pc_id}, {model_id}"

    walks = walks.to(device).requires_grad_(True)
    num_walks = walks.shape[0]
    label = label_tensor.item()
    pc_data = {"vertices": vertices.numpy()}  # For compatibility with old structure

    one_hot = F.one_hot(torch.tensor(label), num_classes=cfg.num_classes).float().to(device)

    # Configs
    w = cfg.attacking_weight
    step_size = cfg.step_size
    max_iter = cfg.max_iter
    max_label_diff = cfg.max_label_diff

    kl_objective = torch.nn.KLDivLoss(reduction="sum")

    num_vertices = pc_data["vertices"].shape[0]
    vertices_counter = np.ones(num_vertices)
    vertices_gradient_change_sum = np.zeros_like(pc_data["vertices"])
    false_classifications = 0
    attack_losses = []

    for iteration in range(max_iter):
        features = walks[:, :, :3].to(torch.float32).to(device).requires_grad_(True)
        vertics_idx = torch.arange(num_vertices)[:features.shape[1]].cpu().numpy()

        #with torch.autograd:
        _, logits = model(features, classify=False, training=False)
        prediction = F.softmax(logits[1], dim=-1)
        source_prediction = prediction.detach().cpu().numpy()[label]

        attack = -1 * w * kl_objective(prediction.log(), one_hot)
        gradients = torch.autograd.grad(outputs=attack, inputs=features, retain_graph=True)[0]
        print(f"Mean gradient norm: {gradients.norm()}")    
        #gradients = torch.autograd.grad(outputs=attack, inputs=features,
        #                                grad_outputs=torch.ones_like(prediction), retain_graph=True)[0]

        attacked_features = features + gradients
        attack_prediction = model(attacked_features, classify=True, training=False)
        attack_prediction = attack_prediction / num_walks
        prediction_abs_diff = abs(source_prediction - attack_prediction[label].item())

        if prediction_abs_diff > cfg.max_label_diff:
            ratio = cfg.max_label_diff / prediction_abs_diff
            gradients = gradients * ratio

        if np.argmax(prediction.detach().cpu().numpy()) != source_prediction:
            false_classifications += 1
        else:
            false_classifications = 0

        attack_losses.append(attack.detach().cpu().numpy())
        vertices_counter[vertics_idx] += 1
        vertices_gradient_change_sum[vertics_idx] += gradients[0].detach().cpu().numpy()

        pc_perturbation = vertices_gradient_change_sum / vertices_counter[:, np.newaxis]
        pc_data["vertices"] += pc_perturbation

        if false_classifications >= 10:
            print(f"Early stopping at {iteration}")
            break

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{model_id}_attacked.npz")
        np.savez(out_path, model_features=walks.detach().cpu().numpy(), label=label, model_id=model_id)
        print(f"Saved attacked walk to: {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/attack_config.json")
    parser.add_argument("--id", type=str, required=True, help="Model ID, e.g., airplane_0123")
    parser.add_argument("--output_dir", type=str, default="attacks")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_dict = json.load(f)
    cfg = SimpleNamespace(**cfg_dict)
    
    walk_dataset = WalksDataset(cfg.walk_npz_root)
    pc_dataset = PointCloudDataset(cfg.original_pc_root)
    attack_single_pc(cfg, args.id, walk_dataset, pc_dataset, output_dir=args.output_dir)