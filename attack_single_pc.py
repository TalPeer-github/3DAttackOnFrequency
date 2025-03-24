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


def attack_single_pc(config, model_id, pc_data=None, output_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attacked_network_params, attacked_dnn_model = define_network_and_its_params(config)
    attacked_dnn_model.to(device)

    # TODO - define dataset
    walk_path = load_walk_npz_by_id(model_id, config["walk_npz_root"])
    data = np.load(walk_path, allow_pickle=True)
    walks = torch.tensor(data["model_features"], dtype=torch.float32)  # (N, T, 3)
    walks = walks.to(device)
    walks.requires_grad = True

    label = int(data["label"])  # Fix
    one_hot = F.one_hot(torch.tensor(label), num_classes=config["num_classes"]).float().to(device)

    w = config.get("attacking_weight", 1.0)
    step_size = config.get("step_size", 0.01)
    max_iter = config.get("max_iter", 50)
    max_label_diff = config.get("max_label_diff", 0.01)

    kl_objective = torch.nn.KLDivLoss(reduction="sum")

    num_vertics = pc_data['vertices'].shape
    vertices_counter = np.ones(num_vertics)
    vertices_gradient_change_sum = np.zeros(num_vertics)
    false_classifications = 0
    attack_losses = []
    for iteration in range(max_iter):
        features, labels = dataset.get_walk_features(pc_data, attacked_network_params)
        features = features[:, :, :3].to(torch.float32).to(device).requires_grad_(True)
        vertics_idx = features[0, :, 3].astype(np.int)

        with torch.autograd:
            prediction = attacked_dnn_model(features, classify=True, training=False)
            attack = -1 * w * kl_objective(one_hot, prediction)
        gradients = torch.autograd.grad(outputs=attack, inputs=features,
                                        grad_outputs=torch.ones_like(pred), retain_graph=True)[0]

        prediction = prediction / args.num_walks
        source_prediction = (prediction.numpy())[config['source_label']]

        attack.backward(features)  # TODO - check if needed (since using gradients above)

        attacked_features = features + gradients
        attack_prediction = attacked_dnn_model(attacked_features, classify=True, training=False)
        attack_prediction = attack_prediction / args.num_walks
        prediction_abs_diff = abs(source_prediction - attack_prediction)

        if prediction_abs_diff > args.max_label_diff:
            ratio = args.max_label_diff / prediction_abs_diff
            gradients = gradients * ratio

        if np.argmax(prediction) != source_prediction:  # TODO - check about source_prediction
            false_classifications += 1
        else:
            false_classifications = 0  # TODO - logic?

        attack_losses.append(attack.numpy())
        vertices_counter[vertics_idx] += 1
        vertices_gradient_change_sum[vertics_idx] += gradients[0].numpy()

        pc_perturbation = vertices_gradient_change_sum / vertices_counter
        pc_data['vertices'] += pc_perturbation

        if num_wrong >= 10:  # Originaly - sanity check if iteration < 15
            print(f"Early stopping at {iteration}")
            # (mesh_data=mesh_data, fileds_needed=fields_needed, out_fn=path[0:-4] + '_attacked.npz')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{model_id}_attacked.npz")
        np.savez(out_path, model_features=walks.cpu().numpy(), label=label, model_id=model_id)
        print(f"Saved attacked walk to: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/attack_config.json")
    parser.add_argument("--id", type=str, required=True, help="Model ID, e.g., airplane_0123")
    parser.add_argument("--output_dir", type=str, default="attacks")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    attack_single_pc(config, args.id, output_dir=args.output_dir)
