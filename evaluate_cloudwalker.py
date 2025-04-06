import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import DataLoader
from dataset import WalksDataset
from cloudwalker_arch import CloudWalkerNet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def custom_collate_fn(batch):
    """Custom collate function for dataloader to handle walks data"""
    walks, labels, ids = zip(*batch)  # list of Tensors and strings
    walks_tensor = torch.stack(walks)   # Assumes all walks have same shape
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return walks_tensor, labels_tensor, list(ids)

def evaluate_cloudwalker(cfg, model_path=None):
    """
    Evaluate CloudWalker model on test set
    
    Args:
        cfg: Configuration object with model and dataset parameters
        model_path: Path to the model checkpoint to evaluate (uses best_model.pth by default)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Setup test dataset and dataloader
    test_dataset = WalksDataset(cfg.test_walk_npz_path)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.args.batch_size, 
        shuffle=False, 
        num_workers=cfg.args.num_workers, 
        collate_fn=custom_collate_fn
    )
    
    # Initialize CloudWalker model
    model = CloudWalkerNet(cfg, cfg.num_classes, net_input_dim=3).to(device)
    
    # Load model weights
    if model_path is None:
        model_path = os.path.join(cfg.logdir, "best_model.pth")
        
    if not os.path.exists(model_path):
        # Try to find the latest checkpoint
        checkpoints = [f for f in os.listdir(cfg.logdir) if f.startswith("learned_model2keep")]
        if checkpoints:
            checkpoints.sort()
            model_path = os.path.join(cfg.logdir, checkpoints[-1])
            print(f"[INFO] Using latest checkpoint: {model_path}")
        else:
            model_path = os.path.join(cfg.logdir, "checkpoint.pth")
            print(f"[INFO] Using checkpoint.pth")
            
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    print(f"[INFO] Loaded model from {model_path}")
    
    # Evaluation
    model.eval()
    correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    
    print(f"[INFO] Evaluating on {len(test_loader)} batches...")
    with torch.no_grad():
        for walk_features, labels, model_ids in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            walk_features = walk_features.to(device)  # [B, num_walks, walk_len, 3]
            labels = labels.to(device)  # [B]
            
            # Get shapes
            batch_size, num_walks, walk_len, _ = walk_features.shape
            
            # Process each walk separately
            batch_preds = []
            
            for w in range(num_walks):
                # Extract the current walk for all batch samples
                walk_batch = walk_features[:, w, :, :]  # [B, walk_len, 3]
                
                # Forward pass
                outputs = model(walk_batch, classify=True)
                
                # Get predictions
                batch_preds.append(outputs)
            
            # Average predictions from all walks for each sample
            avg_probs = torch.stack(batch_preds).mean(dim=0)  # [B, num_classes]
            predictions = torch.argmax(avg_probs, dim=1)  # [B]
            
            # Update metrics
            correct += (predictions == labels).sum().item()
            total_samples += batch_size
            
            # Store predictions and labels for detailed metrics
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate overall accuracy
    accuracy = correct / total_samples
    print(f"[RESULT] Test Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Save results to file
    results_dir = os.path.join(cfg.logdir, "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "evaluation_metrics.txt"), "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        report = classification_report(all_labels, all_preds)
        f.write(report)
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cloudwalker_params.json")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    cli_args = parser.parse_args()

    # Load configuration
    with open(cli_args.config, "r") as f:
        cfg_dict = json.load(f)

    cfg = SimpleNamespace(**cfg_dict)
    cfg.args = SimpleNamespace(**cfg.args)

    # Evaluate model
    evaluate_cloudwalker(cfg, cli_args.model_path)

if __name__ == "__main__":
    main() 