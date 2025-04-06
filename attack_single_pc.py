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
from abc import ABC, abstractmethod
from save_walk_as_npz import generate_random_walks  # Import the walk generation function
import networkx as nx

class PointSelectionStrategy(ABC):
    """Base class for point selection strategies in adversarial attacks."""
    
    @abstractmethod
    def select_points(self, original_pc, walks, model, device):
        """
        Select points to perturb based on the strategy.
        
        Args:
            original_pc: Original point cloud tensor
            walks: Walk tensors
            model: CloudWalker model
            device: Device to run on
            
        Returns:
            mask: Boolean tensor indicating which points to perturb
        """
        pass

class AllPointsStrategy(PointSelectionStrategy):
    """Strategy that selects all points (baseline)."""
    
    def select_points(self, original_pc, walks, model, device):
        return torch.ones(original_pc.shape[0], dtype=torch.bool, device=device)


class MSTStrategy(PointSelectionStrategy):
    def select_points(self, original_pc, walks, model, device):
            N = original_pc.shape[0]
            self.N = N
            cooccur = np.zeros((N, N), dtype=np.int32)

            # Step 1: Build co-occurrence matrix
            for walk in walks:
                for i in range(len(walk)):
                    for j in range(i + 1, len(walk)):
                        a, b = walk[i], walk[j]
                        cooccur[a, b] += 1
                        cooccur[b, a] += 1

            # Step 2: Build full graph with walk-based + high-weight fill-ins
            G = nx.Graph()
            for i in range(N):
                G.add_node(i)
                for j in range(i + 1, N):
                    if cooccur[i, j] > 0:
                        weight = 1.0 / (cooccur[i, j] + 1e-6)
                    else:
                        weight = 1e6  # VERY high weight to "force" inclusion but mark as weak
                    G.add_edge(i, j, weight=weight)

            # Step 3: Compute MST (includes all nodes, penalizes non-walk connections)
            mst = nx.minimum_spanning_tree(G, weight='weight')

            # Step 4: Score nodes by degree in MST
            degrees = dict(mst.degree())
            point_scores = np.array([degrees.get(i, 0) for i in range(N)])

            # Step 5: Select top-K points by centrality
            top_indices = np.argsort(point_scores)[::-1][:self.num_points]
            return top_indices.tolist()



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

def attack_single_pc(cfg, model_id, walk_dataset, pc_dataset, strategy, output_dir=None):
    """
    Attack a single point cloud by perturbing its coordinates using the specified strategy.
    
    Args:
        cfg: Configuration parameters
        model_id: ID of the model to attack
        walk_dataset: Dataset containing walks
        pc_dataset: Dataset containing point clouds
        strategy: PointSelectionStrategy instance
        output_dir: Directory to save results
    """
    print(f"[INFO] Attacking model: {model_id}")
    print(f"[INFO] Using strategy: {strategy.__class__.__name__}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load model
    model = define_network_and_its_params(cfg).to(device)
    model.eval()
    
    # Load original point cloud and walks
    original_pc, original_label, pc_id = pc_dataset.get_by_model_id(model_id)
    original_walks, label_tensor, walk_id = walk_dataset.get_by_model_id(model_id)
    
    # Check for ID consistency
    assert walk_id == pc_id == model_id, f"Mismatch in IDs: {walk_id}, {pc_id}, {model_id}"
    
    # Move data to device
    original_pc = original_pc.to(device)
    original_walks = original_walks.to(device)
    label = label_tensor.item()
    
    # Get original prediction
    original_prediction, original_confidence = get_model_prediction(model, original_walks, device, enable_grad=False)
    print(f"Original prediction: {original_prediction.argmax().item()} (Confidence: {original_confidence:.4f})")
    print(f"True label: {label}")
    
    # Select points to perturb using the strategy
    point_mask = strategy.select_points(original_pc, original_walks, model, device)
    print(f"[INFO] Selected {point_mask.sum().item()} points to perturb")
    
    # Initialize perturbation on the selected points
    delta = torch.zeros_like(original_pc, requires_grad=True, device=device)
    
    # Attack parameters
    epsilon = cfg.attack_epsilon
    iterations = 50  # Reduced from 200 for testing
    required_consecutive_successes = 3  # Number of consecutive successes needed to stop
    min_confidence_threshold = 0.7  # Minimum confidence required for successful attack
    consecutive_successes = 0  # Counter for consecutive successful predictions
    
    # Initialize optimizer with smaller learning rate
    optimizer = torch.optim.Adam([delta], lr=0.001)  # Reduced from default 0.01
    
    # Attack loop variables
    best_loss = float('-inf')
    best_delta = None
    attack_success = False
    last_prediction = None
    
    # Track metrics
    history = {
        'loss': [],
        'confidence': [],
        'prediction': [],
        'strategy': strategy.__class__.__name__
    }
    
    # Attack loop
    for i in tqdm(range(iterations), desc="Attack progress"):
        # Zero gradients
        optimizer.zero_grad()
        
        # Clamp perturbation to ensure it's within epsilon bounds
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            # Only apply perturbations to selected points
            delta.data[~point_mask] = 0
        
        # Apply perturbation to point cloud
        perturbed_pc = original_pc + delta
        
        # Generate new walks for the perturbed point cloud
        # Convert to numpy for walk generation
        perturbed_pc_np = perturbed_pc.detach().cpu().numpy()
        num_walks = original_walks.shape[0]
        seq_len = original_walks.shape[1]
        
        # Generate new walks
        perturbed_walks = generate_random_walks(perturbed_pc_np, num_walks, seq_len)
        perturbed_walks = torch.from_numpy(perturbed_walks).to(device)
        
        # Get model prediction on perturbed walks (enable gradients for attack)
        prediction, confidence = get_model_prediction(model, perturbed_walks, device, enable_grad=True)
        
        # Get probability distribution
        probs = F.softmax(prediction, dim=0)
        
        # Cross-entropy loss with additional diversity term
        prediction = prediction.unsqueeze(0)
        target = torch.tensor([label], device=device)
        ce_loss = F.cross_entropy(prediction, target)
        
        # Add diversity loss to encourage different probability distribution
        diversity_loss = -torch.sum(probs * torch.log(probs + 1e-10))  # Negative entropy
        loss = ce_loss - 0.1 * diversity_loss  # Weighted combination
        
        # Backward pass (maximize loss)
        (-loss).backward()  # Negative for maximization
        
        # Zero out gradients for non-selected points
        delta.grad[~point_mask] = 0
        
        # Update perturbation
        optimizer.step()
        
        # Track metrics
        current_pred = prediction.argmax().item()
        history['loss'].append(loss.item())
        history['confidence'].append(confidence)
        history['prediction'].append(current_pred)
        
        # Check attack status with stricter criteria
        is_different_class = current_pred != label
        is_confident = confidence >= min_confidence_threshold
        prob_difference = torch.abs(probs[current_pred] - probs[label]).item()
        is_significant = prob_difference > 0.3  # Require significant probability shift
        
        if is_different_class and is_confident and is_significant:
            if last_prediction == current_pred:  # Only count if prediction stays the same
                consecutive_successes += 1
                print(f"[SUCCESS] Model fooled with confidence {confidence:.4f}! "
                      f"({consecutive_successes}/{required_consecutive_successes} consecutive times)")
                print(f"Probability difference: {prob_difference:.4f}")
            else:
                consecutive_successes = 1
                print(f"[SUCCESS] Model fooled for the first time with confidence {confidence:.4f}!")
                print(f"Probability difference: {prob_difference:.4f}")
            
            if loss.item() > best_loss:  # Update best delta if loss is better
                best_loss = loss.item()
                best_delta = delta.clone()
                attack_success = True
        else:
            consecutive_successes = 0
            if not is_different_class:
                print(f"[ATTEMPT] Model not fooled yet (predicted class {current_pred})")
            elif not is_confident:
                print(f"[ATTEMPT] Model fooled but confidence too low ({confidence:.4f} < {min_confidence_threshold})")
            else:
                print(f"[ATTEMPT] Model fooled but probability shift too small ({prob_difference:.4f})")
        
        last_prediction = current_pred
        
        # Print detailed progress every 5 iterations
        if (i + 1) % 5 == 0:
            print(f"\nIteration {i+1}/{iterations}:")
            print(f"Loss: {loss.item():.4f}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Current prediction: {current_pred}")
            print(f"Perturbation magnitude: {torch.norm(delta, dim=1).mean().item():.6f}")
            print(f"Probability distribution: {probs.detach().cpu().numpy()}\n")
        
        # Early stopping if we've fooled the model consistently
        if consecutive_successes >= required_consecutive_successes:
            print(f"\n[EARLY STOPPING] Successfully fooled model {consecutive_successes} times in a row!")
            print(f"Final probability distribution: {probs.detach().cpu().numpy()}")
            break

    # Use best delta if available
    if attack_success and best_delta is not None:
        delta = best_delta
    
    # Final evaluation with new walks
    with torch.no_grad():
        perturbed_pc = original_pc + delta
        perturbed_pc_np = perturbed_pc.detach().cpu().numpy()
        final_walks = generate_random_walks(perturbed_pc_np, num_walks, seq_len)
        final_walks = torch.from_numpy(final_walks).to(device)
        final_prediction, final_confidence = get_model_prediction(model, final_walks, device, enable_grad=False)
    
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
        'history': history,
        'strategy': strategy.__class__.__name__,
        'num_points_perturbed': point_mask.sum().item()
    }
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{model_id}_{strategy.__class__.__name__}_attacked.npz")
        
        # Save only perturbed point cloud and attack info
        np.savez(
            out_path,
            vertices=perturbed_pc.detach().cpu().numpy(),
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
    print(f"Strategy: {strategy.__class__.__name__}")
    print(f"Points perturbed: {point_mask.sum().item()}/{original_pc.shape[0]}")
    
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
        strategy = AllPointsStrategy()
        success, perturbation, info = attack_single_pc(
            cfg=cfg,
            model_id=model_id,
            walk_dataset=walk_dataset,
            pc_dataset=pc_dataset,
            strategy=strategy,
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
    parser.add_argument("--strategy", type=str, default="all_points",
                        choices=["all_points", "gradient_magnitude", "graph_centrality", 
                                "mst_based", "walk_intersection", "walk_saliency"],
                        help="Point selection strategy to use")
    
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
    
    # Select strategy
    if args.strategy == "all_points":
        strategy = AllPointsStrategy()
    else:
        raise NotImplementedError(f"Strategy {args.strategy} not implemented yet")
    
    # Run attack
    if args.id:
        # Attack a single model
        walk_dataset = WalksDataset(cfg.walk_npz_root)
        pc_dataset = PointCloudDataset(cfg.original_pc_root)
        attack_single_pc(cfg, args.id, walk_dataset, pc_dataset, strategy, output_dir=cfg.attack_output_dir)
    else:
        # Attack multiple models
        attack_batch(cfg, num_samples=args.num_samples)