import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from victim_models.point_net_pp import PointNetPlusPlus
from attacks.point_cloud_attack import PointCloudAttack, CWAOF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from utils.dataset_utils import create_dataset
from utils.transforms import AddCentralityNodeFeatures, AddWalkFeature, EdgeIndexTransform
from utils.config import args
import os
import open3d as o3d



def visualize_point_cloud(pos,pos_adv,perturbation, true_class="",pred_class="",adv_class="",title_orig="",title_adversarial="",spectral=True):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': '3d'})
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    axes[0].scatter(pos[:, 0], pos[:, 1], pos[:, 2],edgecolor='#670D2F', s=100, alpha=0.2,c='#EF88AD')
    axes[1].scatter(perturbation[:, 0], perturbation[:, 1], perturbation[:, 2],c='grey', s=50)
    axes[2].scatter(pos_adv[:, 0], pos_adv[:, 1], pos_adv[:, 2],c='#C1D8C3',edgecolor='#6A9C89', s=100)
    
    axes[0].set_title(f"Original Point Cloud(True Class:{true_class})\n Clean Prediction: {pred_class}")
    axes[1].set_title(f"Perturbation")
    axes[2].set_title(f"Adversarial Point Cloud: (Attack Prediction: {adv_class})")

    #plt.tight_layout()
    fig_path = f"attacks/visualizations/"
    attack_type = "spectral" if spectral else "noise"
    img_title = f"{fig_path}{attack_type}_{true_class}_adv_{adv_class}.png"
    plt.savefig(img_title)
    plt.show()

def visualize_point_cloud_spectral(pos,pos_adv, true_class="",pred_class="",adv_class="",title_orig="",title_adversarial="",spectral=True):
    def plot_mesh(pos,adv=False):
        points = pos[:, :2].numpy()  # project to XY plane
        tri = Delaunay(points)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        verts = pos[tri.simplices] 

        mesh = Poly3DCollection(verts.numpy(), alpha=0.6, edgecolor='k')
        ax.add_collection3d(mesh)

        ax.set_xlim(pos[:, 0].min(), pos[:, 0].max())
        ax.set_ylim(pos[:, 1].min(), pos[:, 1].max())
        ax.set_zlim(pos[:, 2].min(), pos[:, 2].max())

        fig_path = f"attacks/visualizations/{true_class}/"
        attack_type = "spectral/" if spectral else "noise/"
        exp_name = args.experiment_name
        save_dir = f"{fig_path}{attack_type}{exp_name}"
        os.makedirs(save_dir, exist_ok=True)
        mesh_type = "mesh_adv" if adv else "mesh_clean"
        img_title = f"{save_dir}/{mesh_type}_{pred_class}_adv_{adv_class}.png"
        plt.savefig(img_title)


    true_class = true_class.split(' ')[0]
    plot_mesh(pos,adv=False)
    plot_mesh(pos_adv,adv=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), subplot_kw={'projection': '3d'})

    axes[0].scatter(pos[:, 0], pos[:, 1], pos[:, 2],edgecolor='#670D2F', s=50, alpha=0.5,c='#EF88AD',linewidth=1.25)
    axes[1].scatter(pos_adv[:, 0], pos_adv[:, 1], pos_adv[:, 2],edgecolor='#670D2F', s=50, alpha=0.5,c='#EF88AD',linewidth=1.25)
    axes[0].set_title(f"Clean Prediction: {pred_class}")
    axes[1].set_title(f"Attack Prediction: {adv_class}")

    plt.suptitle(f"Original Point Cloud: {true_class.capitalize()}")

    fig_path = f"attacks/visualizations/{true_class}/"
    attack_type = "spectral/" if spectral else "noise/"
    exp_name = args.experiment_name
    save_dir = f"{fig_path}{attack_type}{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    img_title = f"{save_dir}/clean_{pred_class}_adv_{adv_class}.png"
    plt.savefig(img_title)
    plt.show()

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


def run_experiments(plot=False):
    exp_args = [(0.01,5,500,0.5), (0.01,10,300,0.25), (0.005,10,375,0.5), (0.005,10,450,0.5) ,(0.01,10,500,0.5)] #, (0.0001,5,200), (0.0001,10,200), (0.0001,20,200)]
    best_recorded = (1e-3,10,150,0.5)
    for lr, steps, max_iter, gamma in exp_args:
        args.lr = lr
        args.num_steps = steps #steps
        args.max_iter = max_iter #max_iter
        args.gamma = gamma
        args.experiment_name = f"lr_{args.lr}_steps_{args.num_steps}_iters_{args.max_iter}_gamma_{args.gamma}"
        args.save_dir = f"data/adversarial_examples/{args.experiment_name}"

        start_attack(plot=plot)

def start_attack(plot=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointNetPlusPlus(
        set_abstraction_ratio_1=args.set_abstraction_ratio_1,
        set_abstraction_ratio_2=args.set_abstraction_ratio_2,
        set_abstraction_radius_1=args.set_abstraction_radius_1,
        set_abstraction_radius_2=args.set_abstraction_radius_2,
        dropout=args.dropout).to(args.device)
    
    checkpoint = torch.load("checkpoints/epochs10_90val_acc.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    
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
    
    attack = CWAOF(model, eps=args.epsilon, lr=args.lr, 
                   steps=args.num_steps,device='cuda',
                   max_iter=args.max_iter,gamma=args.gamma,
                   clip_mode='linf')
    
    adversarial_examples, clean_preds, adversarial_preds ,true_labels = attack.batch_attack(test_loader)
    
    true_labels = np.array(true_labels)
    clean_preds = np.array(clean_preds)
    adversarial_preds = np.array(adversarial_preds)
    success_rate = calculate_success_rate(clean_preds, adversarial_preds,true_labels)
    
    if plot:
        for i, adv_example in enumerate(adversarial_examples):

            orig_pos = test_dataset[i].pos.cpu().numpy()
            adv_pos = adv_example.cpu().numpy()

            true_class = f"{args.modelnet10_classes[true_labels[i]]} ({i})"
            pred_class = f"{args.modelnet10_classes[clean_preds[i]]}"
            adv_class = f"{args.modelnet10_classes[adversarial_preds[i]]}"

            title_orig = f"Original Point Cloud (Class: {true_class}) \n\n Model Prediction: {pred_class}"
            title_adversarial_pred = f"Adversarial Prediction (Class: {adv_class})"
        
            visualize_point_cloud_spectral(pos=orig_pos, pos_adv=adv_example, true_class=true_class,pred_class=pred_class,adv_class=adv_class,
                                        title_orig=title_orig,title_adversarial=title_adversarial_pred,spectral=True)

         




if __name__ == "__main__":
    start_attack() 
