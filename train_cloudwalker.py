import os
import json
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler
from dataset import WalksDataset
from cloudwalker_arch import CloudWalkerNet
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

def custom_collate_fn(batch):
    """
    Custom collate function for dataloader to handle walks data
    """
    walks, labels, ids = zip(*batch)  # list of Tensors and strings

    # Stack walks and labels tensors
    walks_tensor = torch.stack(walks)   # Assumes all walks have same shape
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return walks_tensor, labels_tensor, list(ids)

def plot_training_progress(train_losses, val_losses, train_accs, val_accs, save_dir):
    """
    Plot training and validation metrics and save the figures
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_dir: Directory to save the plots
    """
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot losses
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('CloudWalker Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'loss_curves.png'), dpi=300)
    plt.close()
    
    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('CloudWalker Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'accuracy_curves.png'), dpi=300)
    plt.close()
    
    # Combined plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss subplot
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy subplot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'combined_curves.png'), dpi=300)
    plt.close()
    
    print(f"[INFO] Training plots saved to {plots_dir}")

def train_cloudwalker(cfg, args):
    """
    Main training function for CloudWalker network
    """
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Setup datasets and dataloaders
    train_dataset = WalksDataset(cfg.train_walk_npz_path)
    val_dataset = WalksDataset(cfg.test_walk_npz_path)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        collate_fn=custom_collate_fn
    )
    
    # Initialize the CloudWalker model
    cloudwalker = CloudWalkerNet(cfg, cfg.num_classes, net_input_dim=3).to(device)
    
    # Setup optimizer (Adam as used in paper)
    optimizer = torch.optim.Adam(
        cloudwalker.parameters(), 
        lr=args.lr, 
        betas=args.betas, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma
    )
    
    # Loss function - CrossEntropyLoss as per paper for classification
    criterion = torch.nn.CrossEntropyLoss(reduction=args.loss_reduction)
    
    # Training metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0
    start_time = time.time()
    
    # Create a logger for training progress
    log_file = os.path.join(cfg.logdir, "training_log.txt")
    os.makedirs(cfg.logdir, exist_ok=True)
    with open(log_file, "w") as f:
        f.write(f"CloudWalker Training - Started at {datetime.now()}\n")
        f.write(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'Time':<12}\n")
        f.write("-" * 60 + "\n")
    
    # Main training loop
    for epoch in range(1, args.num_epochs + 1):
        epoch_start_time = time.time()
        print(f"\n[INFO] Starting epoch {epoch}/{args.num_epochs}...")
        
        # Training phase
        train_loss, train_accuracy = _train_epoch(
            cloudwalker, train_loader, optimizer, criterion, device, lr_scheduler
        )
        
        # Validation phase
        val_loss, val_accuracy = _validate_epoch(
            cloudwalker, val_loader, criterion, device
        )
        
        # Track elapsed time
        epoch_time = time.time() - epoch_start_time
        
        # Log progress
        with open(log_file, "a") as f:
            f.write(f"{epoch:<6} {train_loss:<12.6f} {train_accuracy:<12.6f} {val_loss:<12.6f} {val_accuracy:<12.6f} {epoch_time:<12.2f}\n")
        
        # Print epoch results
        print(f"[Epoch {epoch:02d}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Plot current progress every 10 epochs
        if epoch % 10 == 0 or epoch == args.num_epochs:
            plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies, cfg.logdir)
        
        # Save model checkpoints
        if epoch % args.save_epochs == 0:
            cloudwalker.save_weights(cfg.logdir, step=epoch, keep=True, optimizer=optimizer)
            print(f"[INFO] Checkpoint saved at epoch {epoch}")
            
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save({"model": cloudwalker.state_dict()}, 
                      os.path.join(cfg.logdir, "best_model.pth"))
            print(f"[INFO] New best validation accuracy: {best_val_acc:.4f}. Model saved.")
    
    # Training complete
    elapsed_time = time.time() - start_time
    print(f"\n[INFO] Training completed in {elapsed_time / 60:.2f} minutes.")
    print(f"[INFO] Best validation accuracy: {best_val_acc:.4f}")
    
    # Final plot of training progress
    plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies, cfg.logdir)
    
    # Save training history
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accuracies,
        "val_acc": val_accuracies,
        "best_val_acc": best_val_acc,
        "total_epochs": args.num_epochs,
        "completed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_time": elapsed_time
    }
    
    with open(os.path.join(cfg.logdir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def _train_epoch(model, dataloader, optimizer, criterion, device, scheduler):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Training")
    
    for walk_features, labels, _ in pbar:
        # Move data to device
        walk_features = walk_features.to(device)  # [B, num_walks, walk_len, 3]
        labels = labels.to(device)  # [B]
        
        # Get shapes
        batch_size, num_walks, walk_len, _ = walk_features.shape
        
        # Process each walk separately and then average the predictions
        all_preds = []
        
        for w in range(num_walks):
            # Extract the current walk for all batch samples
            walk_batch = walk_features[:, w, :, :]  # [B, walk_len, 3]
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass (don't use softmax as CrossEntropyLoss expects logits)
            _, logits = model(walk_batch, classify=False)
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Track statistics
            total_loss += loss.item()
            
            # Store predictions
            all_preds.append(logits)
        
        # Average predictions from all walks for each sample
        avg_logits = torch.stack(all_preds).mean(dim=0)  # [B, num_classes]
        predictions = torch.argmax(avg_logits, dim=1)  # [B]
        
        # Update accuracy metrics
        correct += (predictions == labels).sum().item()
        total_samples += batch_size
        
        # Update progress bar
        pbar.set_postfix({
            "loss": total_loss / (pbar.n + 1),
            "acc": correct / total_samples
        })
    
    # Calculate final metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_samples
    
    return avg_loss, accuracy

def _validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    pbar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for walk_features, labels, _ in pbar:
            # Move data to device
            walk_features = walk_features.to(device)  # [B, num_walks, walk_len, 3]
            labels = labels.to(device)  # [B]
            
            # Get shapes
            batch_size, num_walks, walk_len, _ = walk_features.shape
            
            # Process each walk separately
            all_preds = []
            
            for w in range(num_walks):
                # Extract the current walk for all batch samples
                walk_batch = walk_features[:, w, :, :]  # [B, walk_len, 3]
                
                # Forward pass
                _, logits = model(walk_batch, classify=False)
                
                # Compute loss
                loss = criterion(logits, labels)
                
                # Track statistics
                total_loss += loss.item()
                
                # Store predictions
                all_preds.append(logits)
            
            # Average predictions from all walks for each sample
            avg_logits = torch.stack(all_preds).mean(dim=0)  # [B, num_classes]
            predictions = torch.argmax(avg_logits, dim=1)  # [B]
            
            # Update accuracy metrics
            correct += (predictions == labels).sum().item()
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                "loss": total_loss / (pbar.n + 1),
                "acc": correct / total_samples
            })
    
    # Calculate final metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total_samples
    
    return avg_loss, accuracy

def main():
    """Main entry point for training"""
    print("[INFO] Starting training for CloudWalker network.")
    print("[INFO] Loading configuration...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cloudwalker_params.json")
    parser.add_argument("--plot-only", action="store_true", help="Only plot existing training history without training")
    cli_args = parser.parse_args()

    # Load configuration
    with open(cli_args.config, "r") as f:
        cfg_dict = json.load(f)

    cfg = SimpleNamespace(**cfg_dict)
    cfg.args = SimpleNamespace(**cfg.args)

    # Check if we should only plot existing history
    if cli_args.plot_only:
        history_path = os.path.join(cfg.logdir, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
            
            train_losses = history["train_loss"]
            val_losses = history["val_loss"]
            train_accs = history["train_acc"]
            val_accs = history["val_acc"]
            
            plot_training_progress(train_losses, val_losses, train_accs, val_accs, cfg.logdir)
            print("[INFO] Plotting completed.")
        else:
            print(f"[ERROR] No training history found at {history_path}")
    else:
        print("[INFO] Training initialized. Running epochs...")
        train_cloudwalker(cfg, cfg.args)


if __name__ == "__main__":
    main() 