import argparse
import torch
import os

def get_args():
    parser = argparse.ArgumentParser(description='Point Cloud Attack Configuration')
    
    # Model parameters
    parser.add_argument('--set_abstraction_ratio_1', type=float, default=0.4624,
                       help='First set abstraction ratio')
    parser.add_argument('--set_abstraction_ratio_2', type=float, default=0.2868,
                       help='Second set abstraction ratio')
    parser.add_argument('--set_abstraction_radius_1', type=float, default=0.7562,
                       help='First set abstraction radius')
    parser.add_argument('--set_abstraction_radius_2', type=float, default=0.5,
                       help='Second set abstraction radius')
    parser.add_argument('--dropout', type=float, default=0.18,
                       help='Dropout rate')
    # Dataset parameters
    parser.add_argument('--num_points', type=int, default=2048,
                       help='Number of points to sample')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of neighbors for AddWalkFeature')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    
    parser.add_argument('--num_walks', type=float, default=32)
    parser.add_argument('--walks_len', type=int, default=128)

    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Maximum perturbation allowed')
    
    parser.add_argument('--num_steps', type=int, default=20,
                       help='Number of attack iterations')
    parser.add_argument('--max_iter', type=int, default=175,
                       help='Number of iterations')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.5)

    args = parser.parse_args()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.manual_probs = {1: [1.], 2: [0.5, 0.5], 3: [0.25, 0.5, 0.25], 4: [0.2, 0.3, 0.3, 0.2], 5: [0.15, 0.2, 0.3, 0.2, 0.15],
                6: [0.05, 0.15, 0.3, 0.3, 0.15, 0.05], 7: [0.1, 0.1, 0.15, 0.3, 0.15, 0.1, 0.1],
                8: [0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05],
                9:[0.025,0.025,0.05,0.25,0.3,0.25,0.05,0.025,0.025],
                10:[0.025,0.025,0.05,0.15,0.25,0.25,0.15,0.05,0.025,0.025]}
    

    
    args.modelnet10_labels = {
        'bathtub': 0,
        'bed': 1,
        'chair': 2,
        'desk': 3,
        'dresser': 4,
        'monitor': 5,
        'night_stand': 6,
        'sofa': 7,
        'table': 8,
        'toilet': 9,
    }
    args.modelnet10_classes = {v: k for k, v in args.modelnet10_labels.items()}
    args.filtered_classes = [1, 2, 7, 9]
    args.experiment_name = f"lr_{args.lr}_steps_{args.num_steps}_iters_{args.max_iter}_gamma_{args.gamma}"
    args.save_dir = f"data/adversarial_examples/{args.experiment_name}"
    return args


args = get_args()


default_args = {
    "k": args.k,
    "n_sample": args.num_points,
    "num_walks": args.num_walks,  
    "walks_len": args.walks_len,
    "batch_size": args.batch_size,
    "num_workers": args.num_workers  
}

# ModelNet10 classes and labels (for backward compatibility)
modelnet10_labels = args.modelnet10_labels
modelnet10_classes = args.modelnet10_classes
filtered_classes = args.filtered_classes
manual_probs = args.manual_probs


