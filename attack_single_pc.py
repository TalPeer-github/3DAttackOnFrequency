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
from save_walk_as_npz import generate_random_walks_tensor  # Import the walk generation function
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- STRATEGIES ----------------------------

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
    
class PageRankStrategy(PointSelectionStrategy):
    
    def select_points(self, original_pc, walks, device, model_id=None):
        walks = walks.detach().cpu().numpy().astype(np.int32)
        N = original_pc.shape[0]

        # Step 1: Build co-occurrence graph
        cooccur = np.zeros((N, N), dtype=np.int32)
        for walk in walks:
            for i in range(len(walk)):
                for j in range(i + 1, len(walk)):
                    a, b = walk[i], walk[j]
                    cooccur[a, b] += 1
                    cooccur[b, a] += 1

        # Step 2: Construct graph for PageRank
# Step 2: Construct graph for PageRank
        G = nx.Graph()
        G.add_nodes_from(range(N))  # Ensure all nodes are present even if some are disconnected

        for i in range(N):
            for j in range(i + 1, N):
                if cooccur[i, j] > 0:
                    G.add_edge(i, j, weight=cooccur[i, j])


        assert len(G.nodes) > 0
        # Step 3: Run PageRank
        print("[INFO] Running PageRank...")
        pr_scores = nx.pagerank(G, weight='weight')

        # Step 4: Select top-K points based on PageRank
        sorted_indices = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
        topk = [idx for idx, _ in sorted_indices[:100]]

        # Step 5: Return mask
        mask = np.zeros(N, dtype=bool)
        mask[topk] = True
        cache_path = "cached_pagerank" 
        np.save(cache_path, mask)
        print(f"[INFO] Saved PageRank mask to {cache_path}")
        return torch.tensor(mask, dtype=torch.bool, device=device)

class MSTStrategy(PointSelectionStrategy):
    def __init__(self, cache_dir='cached_msts'):
        super().__init__()
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def select_points(self, original_pc, walks, device, model_id=None):
        """
        Selects important points based on MST of walk co-occurrence, using caching when available.
        """
        if model_id:
            cache_path = os.path.join(self.cache_dir, f"{model_id}_mst_mask.npy")
            if os.path.exists(cache_path):
                print(f"[INFO] Loaded cached MST mask for {model_id}")
                mask_np = np.load(cache_path)
                return torch.tensor(mask_np, dtype=torch.bool, device=device)

        # Compute fresh MST
        walks = walks.detach().cpu().numpy().astype(np.int32)
        N = original_pc.shape[0]
        cooccur = np.zeros((N, N), dtype=np.int32)

        for walk in walks:
            for i in range(len(walk)):
                for j in range(i + 1, len(walk)):
                    a, b = walk[i], walk[j]
                    cooccur[a, b] += 1
                    cooccur[b, a] += 1

        G = nx.Graph()
        for i in range(N):
            G.add_node(i)
            for j in range(i + 1, N):
                weight = 1.0 / (cooccur[i, j] + 1e-6) if cooccur[i, j] > 0 else 1e6
                G.add_edge(i, j, weight=weight)

        mst = nx.minimum_spanning_tree(G, weight='weight')
        print(f"[INFO] MST is Graph with {len(mst.nodes)} nodes and {len(mst.edges)} edges")

        degrees = dict(mst.degree())
        point_scores = np.array([degrees.get(i, 0) for i in range(N)])
        top_indices = np.argsort(point_scores)[::-1][:100]
        point_mask = np.zeros(N, dtype=bool)
        point_mask[top_indices] = True

        cache_path = "cached_msts" 
        np.save(cache_path, point_mask)
        print(f"[INFO] Saved MST mask to {cache_path}")

        return torch.tensor(point_mask, dtype=torch.bool, device=device)


# -------------------- ATTACK_RELATED ----------------------------
def define_network_and_its_params(cfg):
    """
    Loads CloudWalker model from config checkpoint and initializes it.
    """
    # Add logdir parameter to config if not present
    if not hasattr(cfg, 'logdir'):
        cfg.logdir = os.path.dirname(cfg.checkpoint_path)
    
    model = CloudWalkerNet(cfg, cfg.num_classes, net_input_dim=3)
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
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
            walk_batch = walk_batch.to(torch.float)
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
    
    # Get original prediction (logits and confidence)
    original_logits, original_confidence = get_model_prediction(model, original_walks, device, enable_grad=False)
    original_prediction = original_logits.argmax().item()
    temperature = 2.0
    original_probs = F.softmax(original_logits / temperature, dim=0).detach()
    #original_probs = F.softmax(original_logits, dim=0).detach()  # Freeze for KL divergence target
    print(f"Original prediction: {original_prediction} (Confidence: {original_confidence:.4f})")
    print(f"True label: {label}")
    print(f"Original probs: {original_probs}")

    # Select points to perturb using the strategy
    point_mask = strategy.select_points(original_pc, original_walks, device)
    print(f"[INFO] Selected {point_mask.sum().item()} points to perturb")
    
    # Initialize perturbation on the selected points
    delta = torch.zeros_like(original_pc, requires_grad=True, device=device)
    
    # Attack parameters
    epsilon = cfg.attack_epsilon
    iterations = 20  # Reduced from 200 for testing
    optimizer = torch.optim.Adam([delta], lr=0.005)  # Reduced from default 0.01
    
    # Attack loop variables
    best_loss = float('-inf')
    best_delta = None
    attack_success = False
    history = {
        'loss': [],
        'confidence': [],
        'prediction': [],
        'strategy': strategy.__class__.__name__
    }
    num_walks = original_walks.shape[0]
    seq_len = original_walks.shape[1]
    # Attack loop
    for i in tqdm(range(iterations), desc="Attack progress"):
        # Zero gradients
        optimizer.zero_grad()
        
        # Clamp perturbation to ensure it's within epsilon bounds
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data[~point_mask] = 0
        
        # Apply perturbation to point cloud
        perturbed_pc = original_pc + delta
        walk_indices = generate_random_walks_tensor(perturbed_pc, num_walks, seq_len, k_neighbors=cfg.k_neighbors)
        perturbed_walks = perturbed_pc[walk_indices]  # shape: [num_walks, seq_len, 3]    
        prediction, confidence = get_model_prediction(model, perturbed_walks, device, enable_grad=True)
            
        # Compute KL divergence between current and clean prediction
        log_probs = F.log_softmax(prediction, dim=0)
        kl_loss = F.kl_div(log_probs, original_probs, reduction='batchmean')
        print(f"[DEBUG] KL Divergence Loss: {kl_loss:.4f}")
        (-kl_loss).backward()
        print(f"[DEBUG] delta.grad.norm(): {delta.grad.norm():.6f}")
        with torch.no_grad():
            delta.data[~point_mask] = 0

        # Update perturbation
        optimizer.step()
        
        # Track metrics
        current_pred = prediction.argmax().item()
        history['loss'].append(kl_loss.item())
        history['confidence'].append(confidence)
        history['prediction'].append(current_pred)
        
        
        # Print detailed progress every 5 iterations
        if (i + 1) % 5 == 0:
            print(f"\nIteration {i+1}/{iterations}:")
            print(f"Loss: {kl_loss.item():.4f}")
            print(f"Confidence: {confidence:.4f}")
            print(f"Current prediction: {current_pred}")
            print(f"Perturbation magnitude: {torch.norm(delta, dim=1).mean().item():.6f}")
            print(f"Probability distribution: {log_probs.detach().cpu().numpy()}\n")
        

    # Use best delta if available
        if kl_loss.item() > best_loss:
                best_loss = kl_loss.item()
                best_delta = delta.clone()
                attack_success = current_pred != label

    if best_delta is not None:
        delta = best_delta

    with torch.no_grad():
        perturbed_pc = original_pc + delta
        final_indices = generate_random_walks_tensor(perturbed_pc, num_walks, seq_len, k_neighbors=cfg.k_neighbors)
        final_walks = perturbed_pc[final_indices]
        final_prediction, final_confidence = get_model_prediction(model, final_walks, device, enable_grad=False)

    perturbation_magnitude = torch.norm(delta, dim=1).mean().item()

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

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{model_id}_{strategy.__class__.__name__}_attacked.npz")
        np.savez(
            out_path,
            vertices=perturbed_pc.detach().cpu().numpy(),
            label=label,
            model_id=model_id,
            info=info
        )
        print(f"[INFO] Saved attacked data to: {out_path}")

        if cfg.visualize_attacks:
            visualize_attack(original_pc.cpu(), perturbed_pc.detach().cpu(), info, out_path)

    status = "SUCCESS" if attack_success else "FAILED"
    print(f"[ATTACK {status}] Model: {model_id}")
    print(f"Original class: {label} → Final class: {final_prediction.argmax().item()}")
    print(f"Perturbation magnitude: {perturbation_magnitude:.4f}")
    print(f"Strategy: {strategy.__class__.__name__}")
    print(f"Points perturbed: {point_mask.sum().item()}/{original_pc.shape[0]}")

    return attack_success, perturbation_magnitude, info

    print(f"[INFO] Running MST-based heuristic attack on model: {model_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and eval
    model = define_network_and_its_params(cfg).to(device)
    model.eval()

    # Load data
    original_pc, original_label, pc_id = pc_dataset.get_by_model_id(model_id)
    original_walks, label_tensor, walk_id = walk_dataset.get_by_model_id(model_id)
    assert walk_id == pc_id == model_id

    original_pc = original_pc.to(device)
    original_walks = original_walks.to(device)
    label = label_tensor.item()

    # Get initial prediction
    original_logits, original_confidence = get_model_prediction(model, original_walks, device, enable_grad=False)
    original_pred = original_logits.argmax().item()

    print(f"[INFO] Original prediction: {original_pred} (Confidence: {original_confidence:.4f})")
    print(f"[INFO] True label: {label}")

    # Select MST points
    point_mask = strategy.select_points(original_pc, original_walks, device)  # shape: [N]
    print(f"[INFO] Selected {point_mask.sum().item()} points for perturbation")

    # Prepare delta
    delta = torch.zeros_like(original_pc, device=device)

    if shift_type == "centroid":
        centroid = original_pc.mean(dim=0, keepdim=True)
        direction = -F.normalize(original_pc - centroid, dim=1)
        delta[point_mask] = torch.randn_like(original_pc[point_mask]) * alpha

    elif shift_type == "noise":
        noise = F.normalize(torch.randn_like(original_pc), dim=1)
        delta[point_mask] = noise[point_mask] * alpha

    else:
        raise ValueError(f"Unsupported shift_type: {shift_type}")

    # Clamp within epsilon
    delta = torch.clamp(delta, -cfg.attack_epsilon, cfg.attack_epsilon)

    # Apply perturbation
    perturbed_pc = original_pc + delta

    # Generate new walks (fresh, realistic classification)
    num_walks = original_walks.shape[0]
    seq_len = original_walks.shape[1]
    walk_indices = generate_random_walks_tensor(perturbed_pc, num_walks, seq_len, k_neighbors=cfg.k_neighbors)
    perturbed_walks = perturbed_pc[walk_indices]

    final_logits, final_confidence = get_model_prediction(model, perturbed_walks, device, enable_grad=False)
    final_pred = final_logits.argmax().item()

    # Stats
    perturbation_magnitude = torch.norm(delta, dim=1).mean().item()
    attack_success = final_pred != label

    info = {
        'model_id': model_id,
        'original_label': label,
        'original_confidence': original_confidence,
        'final_prediction': final_pred,
        'final_confidence': final_confidence,
        'perturbation_magnitude': perturbation_magnitude,
        'attack_success': attack_success,
        'strategy': strategy.__class__.__name__,
        'shift_type': shift_type,
        'num_points_perturbed': point_mask.sum().item()
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{model_id}_{strategy.__class__.__name__}_{shift_type}_attack.npz")
        np.savez(
            out_path,
            vertices=perturbed_pc.detach().cpu().numpy(),
            label=label,
            model_id=model_id,
            info=info
        )
        print(f"[INFO] Saved attacked point cloud to {out_path}")

        if cfg.visualize_attacks:
            visualize_attack(original_pc.cpu(), perturbed_pc.detach().cpu(), info, out_path)

    # Report
    status = "SUCCESS" if attack_success else "FAILED"
    print(f"[ATTACK {status}] {label} → {final_pred} | Confidence: {final_confidence:.4f} | Magnitude: {perturbation_magnitude:.4f}")

    return attack_success, perturbation_magnitude, info

def select_points_for_pc(model_id, walk_dataset, pc_dataset, strategy):
    print("Generating mask...")
    original_pc, original_label, pc_id = pc_dataset.get_by_model_id(model_id)
    original_walks, label_tensor, walk_id = walk_dataset.get_by_model_id(model_id)
    assert walk_id == pc_id == model_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_pc = original_pc.to(device)
    original_walks = original_walks.to(device)
    
    point_mask = strategy.select_points(original_pc, original_walks, device)  # shape: [N]
    print(f"[INFO] Selected {point_mask.sum().item()} points for perturbation")

def plot_mask_on_pc(original_pc, mask, title="Heuristic Selected Points", save_path="mask.png"):
    """
    Plots the point cloud with MST-selected points highlighted (no graph edges).

    Args:
        original_pc (np.ndarray): [N, 3] array of XYZ coordinates.
        mst_mask (np.ndarray): [N] binary mask, True for MST-selected points.
        title (str): Title of the plot.
        save_path (str): Path to save the resulting PNG image.
    """
    assert len(original_pc) == len(mask), "Point cloud and mask must be same length"

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    # Full point cloud - light gray, low alpha
    ax.scatter(original_pc[:, 0], original_pc[:, 1], original_pc[:, 2],
               s=5, c='lightgray', alpha=0.2, label='Point Cloud')

    # Overlay MST-selected points (yellow)
    selected_points = original_pc[mask]
    ax.scatter(selected_points[:, 0], selected_points[:, 1], selected_points[:, 2],
               s=10, c='pink', alpha=0.4, label='MST-selected', edgecolors='black')

    ax.legend(loc='upper right', fontsize=10)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved MST point visualization to: {save_path}")


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
    parser.add_argument("--strategy", type=str, default="mst",
                        choices=["all_points", "mst", "pagerank"],
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
    elif args.strategy == "mst":
        strategy = MSTStrategy()
    elif args.strategy == "pagerank":
        strategy = PageRankStrategy()
    else:
        raise NotImplementedError(f"Strategy {args.strategy} not implemented yet")
    
    # Run attack
    if args.id:
        # # Attack a single model
        walk_dataset = WalksDataset(cfg.walk_npz_root)
        pc_dataset = PointCloudDataset(cfg.original_pc_root)
        #attack_single_pc(cfg, args.id, walk_dataset, pc_dataset, strategy, output_dir="attacks/cloudwalker")
        original_pc, original_label, pc_id = pc_dataset.get_by_model_id(args.id)
        
        select_points_for_pc(args.id, walk_dataset, pc_dataset, strategy)
        mask_path = "/home/cohen-idan/finalproj/Preprocessing/cached_msts.npy"
        mask = np.load(mask_path)
        plot_mask_on_pc(original_pc, mask)
    else:
        # Attack multiple models
        attack_batch(cfg, num_samples=args.num_samples)
        