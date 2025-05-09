import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import trimesh
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance
from torch_geometric.data import Data

from attacks.point_cloud_attack import CWAOF
from victim_models.point_net_pp import PointNetPlusPlus
from victim_models.mesh_cnn import MeshCNNClassifier
from utils.dataset_utils import create_dataset
from utils.transforms import AddCentralityNodeFeatures, AddWalkFeature, EdgeIndexTransform
from utils.config import args


def calculate_success_rate(original_preds, adversarial_preds,true_labels):
    """Calculate attack success rate"""

    total = len(original_preds)
    model_success = np.sum(original_preds == true_labels).item()
    initial_wrongs = np.sum(true_labels != original_preds).item()
    adv_success = np.sum(true_labels == adversarial_preds).item()
    attack_spesific_success = np.sum((original_preds != adversarial_preds) & (true_labels == original_preds)).item()
    attack_success = np.sum(original_preds != adversarial_preds).item()
    
    total = len(original_preds)
    model_success_rate = model_success / total
    attacked_model_success_rate = adv_success / total
    attack_spesific_success_rate = attack_spesific_success / model_success
    print(f"Clean Model success rate: {model_success_rate:.2%}")
    print(f"Attacked model success rate: {attacked_model_success_rate:.2%}")
    print(f"Attack success rate: {attack_spesific_success_rate:.2%}")
            
    return attack_spesific_success_rate

class MeshMetrics:
    @staticmethod
    def validate_mesh(mesh):
        """
        Validate and repair mesh if necessary
        """
        if not mesh.is_watertight:
            mesh.fill_holes()
        if not mesh.is_winding_consistent:
            mesh.fix_winding()
        return mesh

    @staticmethod
    def hausdorff_distance(mesh1, mesh2):
        """
        Calculate Hausdorff distance between two meshes
        """
        points1 = np.array(mesh1.vertices)
        points2 = np.array(mesh2.vertices)
        
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        dist1, _ = tree1.query(points2)
        dist2, _ = tree2.query(points1)
        
        return max(np.max(dist1), np.max(dist2))

    @staticmethod
    def chamfer_distance(mesh1, mesh2):
        """
        Calculate Chamfer distance between two meshes
        """
        points1 = np.array(mesh1.vertices)
        points2 = np.array(mesh2.vertices)
        
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)
        
        dist1, _ = tree1.query(points2)
        dist2, _ = tree2.query(points1)
        
        return np.mean(dist1) + np.mean(dist2)

    @staticmethod
    def wasserstein_distance(mesh1, mesh2):
        """
        Calculate Wasserstein distance between vertex distributions
        """
        points1 = np.array(mesh1.vertices)
        points2 = np.array(mesh2.vertices)
        
        dist_x = wasserstein_distance(points1[:, 0], points2[:, 0])
        dist_y = wasserstein_distance(points1[:, 1], points2[:, 1])
        dist_z = wasserstein_distance(points1[:, 2], points2[:, 2])
        
        return (dist_x + dist_y + dist_z) / 3

    @staticmethod
    def face_area_ratio(mesh1, mesh2):
        """
        Calculate ratio of face areas
        """
        area1 = np.sum(mesh1.area_faces)
        area2 = np.sum(mesh2.area_faces)
        return area2 / area1

    @staticmethod
    def edge_length_ratio(mesh1, mesh2):
        """
        Calculate ratio of edge lengths
        """
        edges1 = mesh1.edges_unique
        edges2 = mesh2.edges_unique
        
        lengths1 = np.linalg.norm(mesh1.vertices[edges1[:, 0]] - mesh1.vertices[edges1[:, 1]], axis=1)
        lengths2 = np.linalg.norm(mesh2.vertices[edges2[:, 0]] - mesh2.vertices[edges2[:, 1]], axis=1)
        
        return np.mean(lengths2) / np.mean(lengths1)

    @staticmethod
    def self_intersection_ratio(mesh):
        """
        Calculate ratio of self-intersecting faces
        """
        intersections = mesh.self_intersection()
        return len(intersections) / len(mesh.faces)

def load_mesh(mesh_path):
    """
    Load and validate mesh
    """
    mesh = trimesh.load(mesh_path)
    return MeshMetrics.validate_mesh(mesh)

def sample_points_from_mesh(mesh, num_points=2048):
    """
    Sample points from a mesh using Poisson disk sampling.
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def apply_perturbation_to_mesh(mesh, perturbation, epsilon=0.1):
    """Apply point cloud perturbation to mesh vertices"""
    vertices = np.array(mesh.vertices)
    
    tree = cKDTree(vertices)
    _, indices = tree.query(perturbation)
    
    for i, idx in enumerate(indices):
        vertices[idx] += perturbation[i]
    perturbed_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    return MeshMetrics.validate_mesh(perturbed_mesh)

def evaluate_mesh_classifier(mesh, classifier, device):
    """
    Evaluate mesh classifier on a given mesh
    """
    points = np.array(mesh.vertices)
    points = torch.from_numpy(points).float().to(device)
    data = Data(pos=points.unsqueeze(0))
    
    with torch.no_grad():
        pred = classifier(data)
        return pred.argmax(dim=1).item()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    pc_model = PointNetPlusPlus(
    set_abstraction_ratio_1=args.set_abstraction_ratio_1,
    set_abstraction_ratio_2=args.set_abstraction_ratio_2,
    set_abstraction_radius_1=args.set_abstraction_radius_1,
    set_abstraction_radius_2=args.set_abstraction_radius_2,
    dropout=args.dropout).to(args.device)
    
    pc_checkpoint = torch.load("checkpoints/epochs10_90val_acc.pt")
    pc_model.load_state_dict(pc_checkpoint['model_state_dict'])
    pc_model.to(device)
    pc_model.eval()
    
    mesh_model = MeshCNNClassifier()
    mesh_checkpoint = torch.load("victim_models/checkpoints/mesh_cnn_model.pth")
    mesh_model.load_state_dict(mesh_checkpoint['model_state_dict'])
    mesh_model.to(device)
    mesh_model.eval()
    

    
    _, test_dataset = create_dataset(
        dataset_name="ModelNet10",
        name='10',
        num_point_to_sample=args.num_points,
        k = args.k,
        include_train=False,
        include_test=True
    )

    test_dataset = test_dataset[::20]
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers)
    
    attack = CWAOF(pc_model, eps=args.epsilon, lr=args.lr, 
                   steps=args.num_steps,device='cuda',
                   max_iter=args.max_iter,gamma=args.gamma,
                   clip_mode='linf')
    
    adversarial_examples, clean_preds, adversarial_preds ,true_labels = attack.batch_attack(test_loader)
    
    true_labels = np.array(true_labels)
    clean_preds = np.array(clean_preds)
    adversarial_preds = np.array(adversarial_preds)
    success_rate = calculate_success_rate(clean_preds, adversarial_preds,true_labels)
    
    
    mesh_success = 0
    total_samples = 0
    
    hausdorff_dists = []
    chamfer_dists = []
    wasserstein_dists = []
    area_ratios = []
    edge_ratios = []
    self_intersection_ratios = []
    

    for i, adversarial_pc in tqdm(enumerate(adversarial_examples), desc="Evaluating transferability"):


        
   
        # mesh_path = test_dataset.processed_file_names[data.batch[i]]  #TODO 
        perturbation = adversarial_pc.pos.cpu().numpy() - test_loader[i].pos.cpu().numpy() #TODO 
        # original_mesh = load_mesh(mesh_path)   #TODO: rethink 
        perturbed_mesh = apply_perturbation_to_mesh(original_mesh, perturbation)  #TODO 
        
        
        hausdorff_dists.append(MeshMetrics.hausdorff_distance(original_mesh, perturbed_mesh))
        chamfer_dists.append(MeshMetrics.chamfer_distance(original_mesh, perturbed_mesh))
        wasserstein_dists.append(MeshMetrics.wasserstein_distance(original_mesh, perturbed_mesh))
        area_ratios.append(MeshMetrics.face_area_ratio(original_mesh, perturbed_mesh))
        edge_ratios.append(MeshMetrics.edge_length_ratio(original_mesh, perturbed_mesh))
        self_intersection_ratios.append(MeshMetrics.self_intersection_ratio(perturbed_mesh))
        
    
        original_mesh_pred = evaluate_mesh_classifier(original_mesh, mesh_model, device)
        perturbed_mesh_pred = evaluate_mesh_classifier(perturbed_mesh, mesh_model, device)
        
        if original_mesh_pred != perturbed_mesh_pred:
            mesh_success += 1
    
    total_samples = len(adversarial_examples)
    mesh_success_rate = mesh_success / total_samples

    avg_hausdorff = np.mean(hausdorff_dists)
    avg_chamfer = np.mean(chamfer_dists)
    avg_wasserstein = np.mean(wasserstein_dists)
    avg_area_ratio = np.mean(area_ratios)
    avg_edge_ratio = np.mean(edge_ratios)
    avg_self_intersection = np.mean(self_intersection_ratios)


    print("\nAttack Results:")
    print(f"Point Cloud Attack Success Rate: {pc_success_rate:.2%}")
    print(f"Mesh Transferability Success Rate: {mesh_success_rate:.2%}")
    print("\nMesh Quality Metrics:")
    print(f"Average Hausdorff Distance: {avg_hausdorff:.4f}")
    print(f"Average Chamfer Distance: {avg_chamfer:.4f}")
    print(f"Average Wasserstein Distance: {avg_wasserstein:.4f}")
    print(f"Average Face Area Ratio: {avg_area_ratio:.4f}")
    print(f"Average Edge Length Ratio: {avg_edge_ratio:.4f}")
    print(f"Average Self-Intersection Ratio: {avg_self_intersection:.4f}")

if __name__ == "__main__":
    main() 