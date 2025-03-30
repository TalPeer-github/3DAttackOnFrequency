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
    """
    Loads CloudWalker model from config checkpoint and initializes it.
    """
    model = CloudWalkerNet(cfg, cfg.num_classes, net_input_dim=3)
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def perturb_walks(walks, original_pc, delta, device):
    """
    Apply perturbation to walk points by finding closest points in original PC
    and applying the corresponding perturbation.
    
    This implementation intelligently detects whether walk points match original point cloud points:
    1. If points match exactly (which is the common case), it uses direct mapping for efficiency
    2. Only falls back to KD-tree when points don't match exactly (e.g., due to interpolation)
    """
    # Clone walks to avoid modifying the original data
    perturbed_walks = walks.clone()
    
    # Check if the walk points are exactly from the original point cloud
    # by comparing a sample of points (first 10 walk points)
    sample_walks = walks[:min(10, walks.shape[0]), 0].detach().cpu().numpy()
    orig_pc_np = original_pc.detach().cpu().numpy()
    delta_np = delta.detach().cpu().numpy()
    
    # Check if points exactly match (within small epsilon)
    exact_match = True
    for point in sample_walks:
        if not any(np.allclose(point, pc_point, atol=1e-5) for pc_point in orig_pc_np):
            exact_match = False
            break
    
    if exact_match:
        # If walks use exact points from the original point cloud,
        # use a more efficient direct approach
        print("[INFO] Using direct perturbation (walks match original PC points)")
        
        # For each walk point, find the index in the original point cloud
        for w in range(walks.shape[0]):
            for p in range(walks.shape[1]):
                walk_point = walks[w, p].detach().cpu().numpy()
                
                # Find matching point in original PC (first match)
                for idx, pc_point in enumerate(orig_pc_np):
                    if np.allclose(walk_point, pc_point, atol=1e-5):
                        # Apply perturbation directly
                        perturbed_walks[w, p] = walks[w, p] + torch.tensor(delta_np[idx], device=device)
                        break
    else:
        # Use KD-tree for efficient nearest neighbor lookup when points don't match exactly
        print("[INFO] Using KD-tree for perturbation (walks points don't exactly match original PC)")
        kdtree = KDTree(orig_pc_np)
        
        # Process each walk
        for w in range(walks.shape[0]):
            for p in range(walks.shape[1]):
                # Get the walk point
                walk_point = walks[w, p].detach().cpu().numpy()
                
                # Find the nearest point in the original point cloud
                _, idx = kdtree.query(walk_point, k=1)
                
                # Apply the corresponding perturbation to the walk point
                perturbed_walks[w, p] = walks[w, p] + torch.tensor(delta_np[idx], device=device)
    
    return perturbed_walks

def get_model_prediction(model, walks, device):
    """
    Get prediction and confidence from model for a set of walks
    """
    num_walks = walks.shape[0]
    
    # Process each walk and average the predictions
    all_preds = []
    for w in range(num_walks):
        # Extract single walk
        walk_batch = walks[w].unsqueeze(0)  # [1, walk_len, 3]
        
        # Get prediction
        with torch.no_grad():
            _, logits = model(walk_batch, classify=False)
            all_preds.append(logits)
    
    # Average predictions
    avg_logits = torch.cat(all_preds).mean(dim=0)
    probs = F.softmax(avg_logits, dim=0)
    
    # Get confidence
    confidence = probs.max().item()
    
    return avg_logits, confidence

def visualize_attack(original_pc, perturbed_pc, info, save_path=None):
    """Visualize the original and perturbed point clouds"""
    fig = plt.figure(figsize=(15, 10))
    
    # Original point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original_pc[:, 0], original_pc[:, 1], original_pc[:, 2], 
                s=1, c='blue', alpha=0.5)
    ax1.set_title(f"Original: Class {info['original_label']}\nConfidence: {info['original_confidence']:.4f}")
    ax1.set_axis_off()
    
    # Perturbed point cloud
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(perturbed_pc[:, 0], perturbed_pc[:, 1], perturbed_pc[:, 2], 
                s=1, c='red', alpha=0.5)
    ax2.set_title(f"Adversarial: Class {info['final_prediction']}\nConfidence: {info['final_confidence']:.4f}")
    ax2.set_axis_off()
    
    plt.suptitle(f"Adversarial Attack on {info['model_id']}\nPerturbation: {info['perturbation_magnitude']:.4f}", 
                fontsize=16)
    
    # Save figure
    if save_path:
        vis_path = save_path.replace('.npz', '_vis.png')
        plt.savefig(vis_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {vis_path}")
        
    plt.close()
    
    # Plot loss and confidence history
    if 'history' in info and len(info['history']['loss']) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        ax1.plot(info['history']['loss'])
        ax1.set_title('Attack Loss')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Plot confidence
        ax2.plot(info['history']['confidence'])
        ax2.set_title('Model Confidence')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Confidence')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        if save_path:
            hist_path = save_path.replace('.npz', '_history.png')
            plt.savefig(hist_path, dpi=300, bbox_inches='tight')
            print(f"History plots saved to {hist_path}")
        
        plt.close()

def attack_single_pc(cfg, model_id, walk_dataset, pc_dataset, output_dir=None):
    """
    Attack a single point cloud by perturbing its coordinates.
    
    Args:
        cfg: Configuration parameters
        model_id: ID of the model to attack
        walk_dataset: Dataset of pre-computed walks
        pc_dataset: Dataset of original point clouds
        output_dir: Directory to save attack results
    """
    print(f"[INFO] Attacking model: {model_id}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load model
    model = define_network_and_its_params(cfg).to(device)
    model.eval()  # Keep model in evaluation mode for attack

    # Load original point cloud and walks
    original_pc, original_label, pc_id = pc_dataset.get_by_model_id(model_id)
    walks, label_tensor, walk_id = walk_dataset.get_by_model_id(model_id)
    
    # Check for ID consistency
    assert walk_id == pc_id == model_id, f"Mismatch in IDs: {walk_id}, {pc_id}, {model_id}"

    # Move data to device
    original_pc = original_pc.to(device)
    walks = walks.to(device)
    label = label_tensor.item()
    original_label = torch.tensor(label, dtype=torch.long).to(device)
    
    # Get original prediction
    original_prediction, original_confidence = get_model_prediction(model, walks, device)
    print(f"Original prediction: {original_prediction.argmax().item()} (Confidence: {original_confidence:.4f})")
    print(f"True label: {label}")
    
    # Initialize perturbation on the original point cloud
    delta = torch.zeros_like(original_pc, requires_grad=True, device=device)
    
    # Initialize optimizer for perturbation
    optimizer = torch.optim.Adam([delta], lr=cfg.attack_step_size)
    
    # Attack parameters
    epsilon = cfg.attack_epsilon
    iterations = cfg.attack_iterations
    
    # Attack loop variables
    best_loss = float('-inf')
    best_delta = None
    attack_success = False
    
    # Track metrics
    history = {
        'loss': [],
        'confidence': [],
        'prediction': []
    }
    
    # Attack loop
    for i in tqdm(range(iterations), desc="Attack progress"):
        # Zero gradients
        optimizer.zero_grad()
        
        # Apply current perturbation to original PC
        perturbed_pc = original_pc + delta
        
        # Clamp perturbation to ensure it's within epsilon bounds
        with torch.no_grad():
            delta.data = torch.clamp(perturbed_pc - original_pc, -epsilon, epsilon)
            perturbed_pc = original_pc + delta
        
        # Apply the perturbation to the walks
        # This maps each walk point to its closest PC point and applies 
        # the corresponding perturbation
        perturbed_walks = perturb_walks(walks, original_pc, delta, device)
        
        # Get model prediction on perturbed walks
        prediction, confidence = get_model_prediction(model, perturbed_walks, device)
        
        # Cross-entropy loss (negative to maximize)
        attack_loss = -F.cross_entropy(prediction.unsqueeze(0), original_label.unsqueeze(0))
        
        # Compute gradients
        attack_loss.backward()
        
        # Update perturbation
        optimizer.step()

        # Track metrics
        history['loss'].append(attack_loss.item())
        history['confidence'].append(confidence)
        history['prediction'].append(prediction.argmax().item())

        # Get predictions after update
        with torch.no_grad():
            # Check if prediction has changed from original class
            current_pred = prediction.argmax().item()
            if current_pred != original_label.item():
                attack_success = True
                
                # Store best perturbation (higher loss = more confident misclassification)
                if attack_loss.item() < best_loss:
                    best_loss = attack_loss.item()
                    best_delta = delta.clone().detach()
                
                print(f"\nAttack successful at iteration {i+1}!")
                print(f"New prediction: {current_pred} (Confidence: {confidence:.4f})")
        
        # Log progress every 10 iterations
        if (i + 1) % 10 == 0:
            grad_norm = torch.norm(delta.grad.view(-1)).item()
            print(f"Iteration {i+1}/{iterations}, Loss: {attack_loss.item():.4f}, Grad norm: {grad_norm:.4f}")
        
        # Early stopping if attack has been successful for 3 consecutive iterations
        if attack_success and i > 2 and all(p != original_label.item() for p in history['prediction'][-3:]):
            print(f"[INFO] Early stopping at iteration {i+1} - Successfully fooled model")
            break

    # Use best delta if available
    if attack_success and best_delta is not None:
        delta = best_delta
        perturbed_pc = original_pc + delta
        perturbed_walks = perturb_walks(walks, original_pc, delta, device)
        final_prediction, final_confidence = get_model_prediction(model, perturbed_walks, device)
    else:
        final_prediction, final_confidence = prediction, confidence
    
    # Calculate perturbation magnitude
    perturbation_magnitude = torch.norm(delta, dim=1).mean().item()
    
    # Create info dictionary with attack results
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
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{model_id}_attacked.npz")
        
        # Save both perturbed point cloud and perturbed walks
        np.savez(
            out_path,
            vertices=perturbed_pc.detach().cpu().numpy(),
            walks=perturbed_walks.detach().cpu().numpy(),
            original_vertices=original_pc.cpu().numpy(),
            original_walks=walks.cpu().numpy(),
            label=label,
            model_id=model_id,
            info=info
        )
        print(f"[INFO] Saved attacked data to: {out_path}")
        
        # Visualize attack
        if cfg.visualize_attacks:
            visualize_attack(original_pc.cpu(), perturbed_pc.detach().cpu(), info, out_path)
    
    # Print summary
    status = "SUCCESS" if attack_success else "FAILED"
    print(f"[ATTACK {status}] Model: {model_id}")
    print(f"Original class: {label} → Final class: {final_prediction.argmax().item()}")
    print(f"Perturbation magnitude: {perturbation_magnitude:.4f}")
    
    return attack_success, perturbation_magnitude, info


def attack_batch(cfg, model_ids=None, num_samples=None):
    """
    Attack a batch of point clouds
    
    Args:
        cfg: Configuration parameters
        model_ids: List of specific model IDs to attack (if None, will use random samples)
        num_samples: Number of samples to attack (if model_ids is None)
    """
    # Load datasets
    print("[INFO] Loading datasets...")
    pc_dataset = PointCloudDataset(cfg.original_pc_root)
    walk_dataset = WalksDataset(cfg.walk_npz_root)
    
    # Set output directory
    output_dir = cfg.attack_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # If no specific model IDs, select random samples
    if model_ids is None:
        available_ids = list(walk_dataset.id_to_index.keys())
        if num_samples is None:
            num_samples = min(10, len(available_ids))
        model_ids = np.random.choice(available_ids, size=min(num_samples, len(available_ids)), replace=False)
    
    print(f"[INFO] Attacking {len(model_ids)} point cloud models...")
    
    # Track results
    results = []
    
    # Attack each model
    for model_id in model_ids:
        success, perturbation, info = attack_single_pc(
            cfg=cfg,
            model_id=model_id,
            walk_dataset=walk_dataset,
            pc_dataset=pc_dataset,
            output_dir=output_dir
        )
        
        results.append({
            'model_id': model_id,
            'success': success,
            'perturbation': perturbation,
            'original_label': info['original_label'],
            'final_prediction': info['final_prediction']
        })
    
    # Calculate overall success rate
    success_rate = sum(1 for r in results if r['success']) / len(results)
    print(f"\n[RESULTS] Attack success rate: {success_rate:.2%}")
    
    # Save summary
    summary_path = os.path.join(output_dir, "attack_results.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'success_rate': success_rate,
            'results': results
        }, f, indent=2)
    
    print(f"[INFO] Attack results saved to: {summary_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CloudWalker Adversarial Attack")
    parser.add_argument("--config", type=str, default="configs/attack_config.json", 
                        help="Path to attack configuration file")
    parser.add_argument("--id", type=str, required=False, 
                        help="Specific model ID to attack (e.g., airplane_0123)")
    
    # Optional advanced parameters
    advanced_group = parser.add_argument_group('Advanced options')
    advanced_group.add_argument("--num_samples", type=int, 
                        help="Number of samples to attack (only used when --id not specified)")
    advanced_group.add_argument("--epsilon", type=float, 
                        help="Maximum perturbation magnitude (overrides config)")
    advanced_group.add_argument("--iterations", type=int, 
                        help="Number of attack iterations (overrides config)")
    advanced_group.add_argument("--visualize", action="store_true", 
                        help="Visualize attack results")
    
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        cfg_dict = json.load(f)
    cfg = SimpleNamespace(**cfg_dict)
    
    # Override config with command-line arguments
    if args.epsilon is not None:
        cfg.attack_epsilon = args.epsilon
    if args.iterations is not None:
        cfg.attack_iterations = args.iterations
    if args.visualize:
        cfg.visualize_attacks = True
    
    # Run attack
    if args.id:
        # Attack a single model
        walk_dataset = WalksDataset(cfg.walk_npz_root)
        pc_dataset = PointCloudDataset(cfg.original_pc_root)
        attack_single_pc(cfg, args.id, walk_dataset, pc_dataset, output_dir=cfg.attack_output_dir)
    else:
        # Attack multiple models
        attack_batch(cfg, num_samples=args.num_samples)