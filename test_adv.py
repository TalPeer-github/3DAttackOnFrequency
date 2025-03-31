import os
import json
import torch
import numpy as np
from types import SimpleNamespace
from cloudwalker_arch import CloudWalkerNet
import torch.nn.functional as F

def load_config(config_path="configs/attack_config.json"):
    """Load configuration from json file"""
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    return SimpleNamespace(**cfg_dict)

def get_model_prediction(model, walks, device):
    """Get model prediction for a set of walks"""
    num_walks = walks.shape[0]
    all_preds = []
    
    # Process each walk
    for w in range(num_walks):
        walk_batch = walks[w].unsqueeze(0)  # [1, walk_len, 3]
        with torch.no_grad():
            _, logits = model(walk_batch, classify=False)
            all_preds.append(logits)
    
    # Average predictions
    avg_logits = torch.cat(all_preds).mean(dim=0)
    probs = F.softmax(avg_logits, dim=0)
    pred_class = torch.argmax(probs).item()
    confidence = probs[pred_class].item()
    
    return pred_class, confidence, probs.detach().cpu().numpy()

def main():
    # Load configuration
    cfg = load_config()
    
    # Add logdir to config (use the directory of the checkpoint)
    cfg.logdir = os.path.dirname(cfg.checkpoint_path)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # Load model
    model = CloudWalkerNet(cfg, cfg.num_classes, net_input_dim=3)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    # Load adversarial example
    adv_path = "attacks/cloudwalker/airplane_0627_attacked.npz"
    data = np.load(adv_path, allow_pickle=True)
    
    # Extract data
    original_walks = torch.tensor(data['original_walks'], dtype=torch.float32).to(device)
    perturbed_walks = torch.tensor(data['walks'], dtype=torch.float32).to(device)
    true_label = data['label'].item()
    info = data['info'].item()  # Load the info dictionary
    
    # Get predictions for original and perturbed walks
    print("\nEvaluating Original Point Cloud:")
    orig_pred, orig_conf, orig_probs = get_model_prediction(model, original_walks, device)
    print(f"True Label: {true_label}")
    print(f"Prediction: {orig_pred} (Confidence: {orig_conf:.4f})")
    
    print("\nEvaluating Adversarial Point Cloud:")
    adv_pred, adv_conf, adv_probs = get_model_prediction(model, perturbed_walks, device)
    print(f"Prediction: {adv_pred} (Confidence: {adv_conf:.4f})")
    
    # Print attack details from saved info
    print("\nAttack Details:")
    print(f"Attack {'succeeded' if info['attack_success'] else 'failed'}")
    print(f"Perturbation Magnitude: {info['perturbation_magnitude']:.4f}")
    print(f"Original Confidence: {info['original_confidence']:.4f}")
    print(f"Final Confidence: {info['final_confidence']:.4f}")
    
    # Print probability distribution for all classes
    print("\nProbability Distribution:")
    for i, (orig_p, adv_p) in enumerate(zip(orig_probs, adv_probs)):
        print(f"Class {i}: Original: {orig_p:.4f}, Adversarial: {adv_p:.4f}")

if __name__ == "__main__":
    main()
