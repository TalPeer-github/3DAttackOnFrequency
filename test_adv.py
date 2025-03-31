import os
import json
import torch
import numpy as np
from types import SimpleNamespace
import torch.nn.functional as F
from cloudwalker_arch import CloudWalkerNet

def load_config(config_path="configs/attack_config.json"):
    """Load and return configuration"""
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    cfg = SimpleNamespace(**cfg_dict)
    # Add logdir attribute (required by CloudWalkerNet)
    cfg.logdir = os.path.dirname(cfg.checkpoint_path)
    return cfg

def get_model_prediction(model, walks, device):
    """
    Get detailed prediction from model for a set of walks
    Returns prediction, confidence, and full probability distribution
    """
    model.eval()
    num_walks = walks.shape[0]
    all_preds = []
    
    with torch.no_grad():
        for w in range(num_walks):
            # Process single walk
            walk_batch = walks[w].unsqueeze(0)  # [1, walk_len, 3]
            _, logits = model(walk_batch, classify=False)
            all_preds.append(logits)
        
        # Average predictions
        avg_logits = torch.cat(all_preds).mean(dim=0)
        probs = F.softmax(avg_logits, dim=0)
        
        # Get prediction and confidence
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()
        
        return pred_class, confidence, probs

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    cfg = load_config()
    
    # Load model
    model = CloudWalkerNet(cfg, cfg.num_classes, net_input_dim=3).to(device)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Loaded model from: {cfg.checkpoint_path}")
    
    # Load adversarial example
    adv_path = "attacks/cloudwalker/airplane_0630_attacked.npz"
    print(f"Loading adversarial example from: {adv_path}")
    
    data = np.load(adv_path, allow_pickle=True)
    original_walks = torch.FloatTensor(data['original_walks']).to(device)
    perturbed_walks = torch.FloatTensor(data['walks']).to(device)
    true_label = int(data['label'])
    info = data['info'].item()
    
    print("\nEvaluating Original Point Cloud:")
    print("---------------------------------")
    orig_pred, orig_conf, orig_probs = get_model_prediction(model, original_walks, device)
    print(f"True Label: {true_label}")
    print(f"Prediction: {orig_pred} (Confidence: {orig_conf:.4f})")
    print("\nOriginal Probability Distribution:")
    for i, prob in enumerate(orig_probs.cpu().numpy()):
        print(f"Class {i}: {prob:.4f}")
    
    print("\nEvaluating Adversarial Point Cloud:")
    print("-----------------------------------")
    adv_pred, adv_conf, adv_probs = get_model_prediction(model, perturbed_walks, device)
    print(f"Prediction: {adv_pred} (Confidence: {adv_conf:.4f})")
    print("\nAdversarial Probability Distribution:")
    for i, prob in enumerate(adv_probs.cpu().numpy()):
        print(f"Class {i}: {prob:.4f}")
    
    print("\nAttack Details:")
    print("--------------")
    print(f"Attack {'succeeded' if adv_pred != true_label else 'failed'}")
    print(f"Perturbation Magnitude: {info['perturbation_magnitude']:.4f}")
    
    # Compare probability shifts
    print("\nProbability Shifts:")
    print("------------------")
    for i in range(cfg.num_classes):
        shift = adv_probs[i].item() - orig_probs[i].item()
        print(f"Class {i}: {shift:+.4f}")

if __name__ == "__main__":
    main()
