import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from cloudwalker_arch import CloudWalkerNet
from dataset import PointCloudDataset, WalksDataset

def define_network_and_its_params(cfg):
    if not hasattr(cfg, 'logdir'):
        cfg.logdir = os.path.dirname(cfg.checkpoint_path)
    model = CloudWalkerNet(cfg, cfg.num_classes, net_input_dim=3)
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def perturb_walks(walks, original_pc, delta, device):
    perturbed_walks = walks.clone()
    sample_walks = walks[:min(10, walks.shape[0]), 0].detach().cpu().numpy()
    orig_pc_np = original_pc.detach().cpu().numpy()

    exact_match = all(
        any(np.allclose(point, pc_point, atol=1e-5) for pc_point in orig_pc_np)
        for point in sample_walks
    )

    if exact_match:
        print("[INFO] Using direct perturbation (walks match original PC points)")
        for w in range(walks.shape[0]):
            for p in range(walks.shape[1]):
                walk_point = walks[w, p].detach().cpu().numpy()
                for idx, pc_point in enumerate(orig_pc_np):
                    if np.allclose(walk_point, pc_point, atol=1e-5):
                        perturbed_walks[w, p] = walks[w, p] + delta[idx]
                        break
    else:
        print("[INFO] Using KD-tree for perturbation (walks points don't exactly match original PC)")
        kdtree = KDTree(orig_pc_np)
        for w in range(walks.shape[0]):
            for p in range(walks.shape[1]):
                walk_point = walks[w, p].detach().cpu().numpy()
                _, idx = kdtree.query(walk_point, k=1)
                if idx < delta.shape[0]:
                    perturbed_walks[w, p] = walks[w, p] + delta[idx]
    return perturbed_walks

def get_model_prediction(model, walks, device, enable_grad=False):
    was_training = model.training
    if enable_grad:
        model.train()
    try:
        with torch.set_grad_enabled(enable_grad):
            _, logits = model(walks, classify=False)
        avg_logits = logits.mean(dim=0)
        probs = F.softmax(avg_logits, dim=0)
        confidence = probs.max().item()
        return avg_logits, confidence
    finally:
        model.train(was_training)

def visualize_attack(original_pc, perturbed_pc, info, save_path=None):
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_pc[:, 0], original_pc[:, 1], original_pc[:, 2], s=1, c='blue', alpha=0.5)
    ax1.set_title(f"Original: Class {info['original_label']}\nConfidence: {info['original_confidence']:.4f}")
    ax1.set_axis_off()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(perturbed_pc[:, 0], perturbed_pc[:, 1], perturbed_pc[:, 2], s=1, c='red', alpha=0.5)
    ax2.set_title(f"Adversarial: Class {info['final_prediction']}\nConfidence: {info['final_confidence']:.4f}")
    ax2.set_axis_off()

    plt.suptitle(f"Adversarial Attack on {info['model_id']}\nPerturbation: {info['perturbation_magnitude']:.4f}", fontsize=16)
    if save_path:
        vis_path = save_path.replace('.npz', '_vis.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {vis_path}")
    plt.close()

    if 'history' in info and len(info['history']['loss']) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(info['history']['loss'])
        ax1.set_title('Attack Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        ax2.plot(info['history']['confidence'])
        ax2.set_title('Model Confidence')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Confidence')
        ax2.grid(True)
        plt.tight_layout()
        if save_path:
            hist_path = save_path.replace('.npz', '_history.png')
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            print(f"History plots saved to {hist_path}")
        plt.close()

def attack_single_pc(cfg, model_id, walk_dataset, pc_dataset, output_dir=None):
    print(f"[INFO] Attacking model: {model_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    model = define_network_and_its_params(cfg).to(device)

    original_pc, original_label, pc_id = pc_dataset.get_by_model_id(model_id)
    walks, label_tensor, walk_id = walk_dataset.get_by_model_id(model_id)
    assert walk_id == pc_id == model_id

    original_pc = original_pc.to(device)
    walks = walks.to(device)
    label = label_tensor.item()
    original_label = torch.tensor(label, dtype=torch.long).to(device)

    original_prediction, original_confidence = get_model_prediction(model, walks, device, enable_grad=False)
    print(f"Original prediction: {original_prediction.argmax().item()} (Confidence: {original_confidence:.4f})")
    print(f"True label: {label}")

    delta = torch.zeros_like(original_pc, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=cfg.attack_step_size)
    epsilon = cfg.attack_epsilon
    iterations = cfg.attack_iterations

    best_loss = float('-inf')
    best_delta = None
    attack_success = False
    history = {'loss': [], 'confidence': [], 'prediction': []}

    for i in tqdm(range(iterations), desc="Attack progress"):
        optimizer.zero_grad()
        perturbed_pc = original_pc + delta
        delta.data.copy_((perturbed_pc - original_pc).clamp(-epsilon, epsilon))
        perturbed_pc = original_pc + delta

        perturbed_walks = perturb_walks(walks, original_pc, delta, device)
        prediction, confidence = get_model_prediction(model, perturbed_walks, device, enable_grad=True)

        if prediction.argmax().item() != label:
            attack_success = True
            print(f"[INFO] Attack succeeded at iteration {i}")
            break

        prediction = prediction.unsqueeze(0)
        target = torch.tensor([label], device=device)
        loss = F.cross_entropy(prediction, target)
        loss.backward()
        optimizer.step()

        history['loss'].append(loss.item())
        history['confidence'].append(confidence)
        history['prediction'].append(prediction.argmax().item())

        if loss.item() > best_loss:
            best_loss = loss.item()
            best_delta = delta.clone()

        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{iterations}, Loss: {loss.item():.4f}, Confidence: {confidence:.4f}, Predicted: {prediction.argmax().item()}")

    if attack_success and best_delta is not None:
        delta = best_delta
        perturbed_pc = original_pc + delta
        perturbed_walks = perturb_walks(walks, original_pc, delta, device)
        final_prediction, final_confidence = get_model_prediction(model, perturbed_walks, device, enable_grad=True)
    else:
        final_prediction, final_confidence = prediction.squeeze(0), confidence

    perturbation_magnitude = torch.norm(delta, dim=1).mean().item()
    info = {
        'model_id': model_id,
        'original_label': label,
        'original_confidence': original_confidence,
        'final_prediction': final_prediction.argmax().item(),
        'final_confidence': final_confidence,
        'perturbation_magnitude': perturbation_magnitude,
        'attack_success': attack_success,
        'history': history
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{model_id}_attacked.npz")
        np.savez(out_path,
                 vertices=perturbed_pc.detach().cpu().numpy(),
                 walks=perturbed_walks.detach().cpu().numpy(),
                 original_vertices=original_pc.cpu().numpy(),
                 original_walks=walks.cpu().numpy(),
                 label=label,
                 model_id=model_id,
                 info=info)
        print(f"[INFO] Saved attacked data to: {out_path}")
        if cfg.visualize_attacks:
            visualize_attack(original_pc.cpu(), perturbed_pc.detach().cpu(), info, out_path)

    print(f"[ATTACK {'SUCCESS' if attack_success else 'FAILED'}] Model: {model_id}")
    print(f"Original class: {label} → Final class: {final_prediction.argmax().item()}")
    print(f"Perturbation magnitude: {perturbation_magnitude:.4f}")
    return attack_success, perturbation_magnitude, info
