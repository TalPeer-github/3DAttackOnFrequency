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
    # Add logdir parameter to config if not present
    if not hasattr(cfg, 'logdir'):
        cfg.logdir = os.path.dirname(cfg.checkpoint_path)
    
    model = CloudWalkerNet(cfg, cfg.num_classes, net_input_dim=3)
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model

def perturb_walks(walks, original_pc, delta, device):
    """
    Apply perturbation to walk points by finding closest points in original PC
    and applying the corresponding perturbation.
    Returns perturbed walks with proper gradient flow.
    """
    # Clone walks to avoid modifying the original data
    perturbed_walks = walks.clone()
    
    # Process walks in batches to avoid memory issues
    batch_size = 1000  # Adjust based on available memory
    
    for start_idx in range(0, walks.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, walks.shape[0])
        batch_walks = walks[start_idx:end_idx]
        
        # Compute pairwise distances between walk points and original PC points
        # [batch_size * walk_len, 1, 3] - [1, num_points, 3] -> [batch_size * walk_len, num_points]
        walk_points = batch_walks.reshape(-1, 1, 3)
        pc_points = original_pc.unsqueeze(0)
        
        distances = torch.norm(walk_points - pc_points, dim=2)
        
        # Find nearest neighbors
        nearest_indices = torch.argmin(distances, dim=1)
        
        # Reshape indices to match batch shape
        nearest_indices = nearest_indices.reshape(end_idx - start_idx, batch_walks.shape[1])
        
        # Apply perturbations using gathered indices
        for w in range(end_idx - start_idx):
            for p in range(batch_walks.shape[1]):
                idx = nearest_indices[w, p]
                perturbed_walks[start_idx + w, p] = walks[start_idx + w, p] + delta[idx]
    
    return perturbed_walks

def get_model_prediction(model, walks, device, enable_grad=False):
    """
    Get prediction and confidence from model for a set of walks
    
    Args:
        model: The CloudWalker model
        walks: The walk tensors
        device: The device to run on
        enable_grad: Whether to enable gradient computation
    """
    num_walks = walks.shape[0]
    
    # Process each walk and average the predictions
    all_preds = []
    
    # Store original training mode
    was_training = model.training
    
    # Set to training mode if we need gradients (for CUDNN RNN backward compatibility)
    if enable_grad:
        model.train()
    
    try:
        for w in range(num_walks):
            # Extract single walk
            walk_batch = walks[w].unsqueeze(0)  # [1, walk_len, 3]
            
            # Get prediction
            if enable_grad:
                _, logits = model(walk_batch, classify=False)
            else:
                with torch.no_grad():
                    _, logits = model(walk_batch, classify=False)
            all_preds.append(logits)
        
        # Average predictions
        avg_logits = torch.cat(all_preds).mean(dim=0)
        probs = F.softmax(avg_logits, dim=0)
        
        # Get confidence
        confidence = probs.max().item()
        
        return avg_logits, confidence
    
    finally:
        # Restore original training mode
        model.train(was_training)

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
    """
    print(f"[INFO] Attacking model: {model_id}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load model
    model = define_network_and_its_params(cfg).to(device)
    model.eval()
    
    # Load original point cloud and walks
    original_pc, original_label, pc_id = pc_dataset.get_by_model_id(model_id)
    walks, label_tensor, walk_id = walk_dataset.get_by_model_id(model_id)
    
    # Check for ID consistency
    assert walk_id == pc_id == model_id, f"Mismatch in IDs: {walk_id}, {pc_id}, {model_id}"
    
    # Move data to device
    original_pc = original_pc.to(device)
    walks = walks.to(device)
    label = label_tensor.item()
    
    # Get original prediction
    original_prediction, original_confidence = get_model_prediction(model, walks, device, enable_grad=False)
    print(f"Original prediction: {original_prediction.argmax().item()} (Confidence: {original_confidence:.4f})")
    print(f"True label: {label}")
    
    # Initialize perturbation on the original point cloud
    delta = torch.zeros_like(original_pc, requires_grad=True, device=device)
    
    # Initialize optimizer for perturbation
    optimizer = torch.optim.Adam([delta], lr=cfg.attack_step_size)
    
    # Attack parameters
    epsilon = cfg.attack_epsilon
    iterations = 20  # Reduced from 200 for testing
    required_consecutive_successes = 3  # Number of consecutive successes needed to stop
    consecutive_successes = 0  # Counter for consecutive successful predictions
    
    # Attack loop variables
    best_loss = float('-inf')
    best_delta = None
    attack_success = False
    last_prediction = None
    
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
        
        # Clamp perturbation to ensure it's within epsilon bounds
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        
        # Apply the perturbation to the walks
        perturbed_walks = perturb_walks(walks, original_pc, delta, device)
        
        # Get model prediction on perturbed walks (enable gradients for attack)
        prediction, confidence = get_model_prediction(model, perturbed_walks, device, enable_grad=True)
        
        # Cross-entropy loss
        prediction = prediction.unsqueeze(0)
        target = torch.tensor([label], device=device)
        loss = F.cross_entropy(prediction, target)  # Regular cross-entropy (we'll maximize it)
        
        # Backward pass (maximize loss)
        (-loss).backward()  # Negative for maximization
        
        # Update perturbation
        optimizer.step()
        
        # Track metrics
        current_pred = prediction.argmax().item()
        history['loss'].append(loss.item())
        history['confidence'].append(confidence)
        history['prediction'].append(current_pred)
        
        # Check attack status
        if current_pred != label:
            if last_prediction != label:  # If last prediction was also wrong
                consecutive_successes += 1
                print(f"[SUCCESS] Model fooled! ({consecutive_successes}/{required_consecutive_successes} consecutive times)")
            else:
                consecutive_successes = 1
                print(f"[SUCCESS] Model fooled for the first time in this sequence!")
            
            if loss.item() > best_loss:  # Update best delta if loss is better
                best_loss = loss.item()
                best_delta = delta.clone()
                attack_success = True
        else:
            consecutive_successes = 0
            print(f"[ATTEMPT] Model not fooled yet (predicted class {current_pred})")
        
        last_prediction = current_pred
        
        # Print detailed progress every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"\nIteration {i+1}/{iterations}:")
            print(f"Loss: {loss.item():.4f}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Current prediction: {current_pred}")
            print(f"Perturbation magnitude: {torch.norm(delta, dim=1).mean().item():.6f}\n")
        
        # Early stopping if we've fooled the model consistently
        if consecutive_successes >= required_consecutive_successes:
            print(f"\n[EARLY STOPPING] Successfully fooled model {consecutive_successes} times in a row!")
            break

    # Use best delta if available
    if attack_success and best_delta is not None:
        delta = best_delta
    
    # Final evaluation
    with torch.no_grad():
        perturbed_pc = original_pc + delta
        perturbed_walks = perturb_walks(walks, original_pc, delta, device)
        final_prediction, final_confidence = get_model_prediction(model, perturbed_walks, device, enable_grad=False)
    
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