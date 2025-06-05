import os
import math
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from config import get_train_config, get_test_config
from data import ModelNet10
from mesh_layers import MeshNet
from utils.retrival import append_feature, calculate_map

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_to_class = {
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

class_to_label = {
    0:'bathtub',
    1:'bed',
    2:'chair',
    3:'desk',
    4:'dresser',
    5:'monitor',
    6:'night_stand',
    7:'sofa',
    8:'table',
    9:'toilet',
}
    
class_names = [class_to_label[i] for i in range(len(class_to_label))]

def start_train():
    def train_model(model, criterion, optimizer, scheduler, cfg):

        best_acc = 0.0
        best_map = 0.0
        best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(1, cfg['max_epoch']):

            print('-' * 60)
            print('Epoch: {} / {}'.format(epoch, cfg['max_epoch']))
            print('-' * 60)

            for phrase in ['train', 'test']:

                if phrase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                ft_all, lbl_all = None, None

                for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader[phrase]):
                    centers = centers.to(device)
                    corners = corners.to(device)
                    normals = normals.to(device)
                    neighbor_index = neighbor_index.to(device)
                    targets = targets.to(device)

                    with torch.set_grad_enabled(phrase == 'train'):
                        outputs, feas = model(centers, corners, normals, neighbor_index)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, targets)

                        if phrase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        if phrase == 'test' and cfg['retrieval_on']:
                            ft_all = append_feature(ft_all, feas.detach().cpu())
                            lbl_all = append_feature(lbl_all, targets.detach().cpu(), flaten=True)

                        running_loss += loss.item() * centers.size(0)
                        running_corrects += torch.sum(preds == targets.data)

                epoch_loss = running_loss / len(data_set[phrase])
                epoch_acc = running_corrects.double() / len(data_set[phrase])

                if phrase == 'train':
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phrase, epoch_loss, epoch_acc))
                    scheduler.step()

                if phrase == 'test':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                    print_info = '{} Loss: {:.4f} Acc: {:.4f} (best {:.4f})'.format(phrase, epoch_loss, epoch_acc, best_acc)

                    if cfg['retrieval_on']:
                        epoch_map = calculate_map(ft_all, lbl_all)
                        if epoch_map > best_map:
                            best_map = epoch_map
                        print_info += ' mAP: {:.4f}'.format(epoch_map)
                    
                    if epoch % cfg['save_steps'] == 0:
                        torch.save(copy.deepcopy(model.state_dict()), os.path.join(cfg['ckpt_root'], '{}.pkl'.format(epoch)))
                    
                    print(print_info)

        print('Best val acc: {:.4f}'.format(best_acc))
        print('Config: {}'.format(cfg))

        return best_model_wts

    cfg = get_train_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
    
    seed = cfg['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 


    data_set = {x: ModelNet10(cfg=cfg['dataset'], part=x) 
                for x in ['train', 'test']}   
    
    data_loader = {x: data.DataLoader(data_set[x], 
                       batch_size=cfg['batch_size'], 
                       num_workers=4, 
                       shuffle=True, 
                       pin_memory=False)
                    for x in ['train', 'test']}

    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    model.to(device)
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()


    if cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    
    if cfg['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['max_epoch'])

    if not os.path.exists(cfg['ckpt_root']):
        os.mkdir(cfg['ckpt_root'])
    best_model_wts = train_model(model, criterion, optimizer, scheduler, cfg)
    torch.save(best_model_wts, os.path.join(cfg['ckpt_root'], 'MeshNet_best.pkl'))


def plot_mesh_from_arrays(vertices, faces, targets,preds,model_type, save_path=None):
    faces = faces.astype(int)
    corners = vertices[faces]  
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(corners, alpha=0.7, edgecolor='k', linewidth=0.3)
    ax.add_collection3d(mesh)

    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
               c='red', s=8, alpha=0.6, label='Vertices')

    max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max() / 2
    mid = vertices.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    fig_title = f"True label = {targets} | Predition = {preds}\n({model_type})"
    ax.set_title(fig_title)
    
    plt.tight_layout()
    fig_path = f"../attacks/visualizations/{model_type}"
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(f"{fig_path}/true_{targets}_pred_{preds}.png")


def plot_confusion_matrix(csv_path, save_to_path): 
    df = pd.read_csv(csv_path)
    y_clean = df["clean"].astype(int)
    y_adv = df["adv"].astype(int)
    
    cm = confusion_matrix(y_clean, y_adv)
    class_names = [class_to_label[i] for i in range(len(class_to_label))]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Clean vs. Adversarial Predictions')
    ax.set_xlabel('Adversarial MeshNet Label')
    ax.set_ylabel('Clean MeshNet Label')

    plt.tight_layout()
    plt.savefig(f"{save_to_path}.png")
    plt.close()


def create_report(csv_path, model="MeshNet"):
    df = pd.read_csv(csv_path)
    y_true = df["true"].astype(int)
    y_clean = df["clean"].astype(int)
    y_adv = df["adv"].astype(int)
    
    clean_acc = accuracy_score(y_true, y_clean)
    clean_prec = precision_score(y_true, y_clean, average='weighted', zero_division=0)
    clean_rec = recall_score(y_true, y_clean, average='weighted', zero_division=0)
    clean_f1 = f1_score(y_true, y_clean, average='weighted', zero_division=0)
    clean_report = classification_report(y_true, y_clean, target_names=class_names, zero_division=0)

    print(f"Clean {model}:")
    print(clean_report)

    adv_acc = accuracy_score(y_true, y_adv)
    adv_prec = precision_score(y_true, y_adv, average='weighted', zero_division=0)
    adv_rec = recall_score(y_true, y_adv, average='weighted', zero_division=0)
    adv_f1 = f1_score(y_true, y_adv, average='weighted', zero_division=0)
    adv_report = classification_report(y_true, y_adv, target_names=class_names, zero_division=0)
    
    print(f"Attacked {model}:")
    print(adv_report)



def plot_confusion_matrix_comparison(csv_path,save_to_path,model='MeshNet'):
    df = pd.read_csv(csv_path)
    y_true = df["true"].astype(int)
    y_clean = df["clean"].astype(int)
    y_adv = df["adv"].astype(int)
    cm_clean = confusion_matrix(y_true, y_clean)
    cm_adv = confusion_matrix(y_true, y_adv)


    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(cm_clean, annot=True, fmt='d', cmap='Blues', linewidth=.5,
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Clean Predictions')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    sns.heatmap(cm_adv, annot=True, fmt='d', cmap='Blues', linewidth=.5,
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Attacked Predictions')
    axes[1].set_xlabel('Adversarial Label')
    axes[1].set_ylabel('True Label')

    plt.suptitle(f'Clean vs. Adversarial - {model}',fontweight='bold', fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{save_to_path}_comp.png")


def plot_mesh(centers, corners, normals,targets,preds,model_type):
    """
    Visualizes a mesh from ModelNet10 dataset format.
    
    Args:
        centers: Tensor of shape (3, N) containing face centers
        corners: Tensor of shape (9, N) containing face corners (3 vertices per face)
        normals: Tensor of shape (3, N) containing face normals
        title: Optional title for the plot
    """
    centers = centers.cpu().numpy()
    corners = corners.cpu().numpy()
    normals = normals.cpu().numpy()

    targets = class_to_label[targets.cpu().numpy().item()]
    preds = class_to_label[preds.cpu().numpy().item()]

    corners = corners.T.reshape(-1, 3, 3)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    mesh = Poly3DCollection(corners, alpha=0.7, linewidth=0.3)
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.scatter(centers[0,:], centers[1,:], centers[2,:], c='r', s=10, alpha=0.5)
    
    for i in range(len(centers[0])):
         ax.quiver(centers[0,:][i], centers[1,:][i], centers[2,:][i],
                  normals[0][i], normals[1][i], normals[2][i],
                  length=0.1, color='b', alpha=0.3)
    

    max_range = max(
         (corners.max(axis=(0,1)) - corners.min(axis=(0,1))).max(),
         (centers.max(axis=1) - centers.min(axis=1)).max()
     ) / 2.0
    
    mid = centers.mean(axis=1)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_box_aspect([1, 1, 1])
    

    fig_title = f"True label = {targets} | Predition = {preds}\n({model_type})"
    ax.set_title(fig_title)
    
    plt.tight_layout()
    fig_path = f"../attacks/visualizations/{model_type}"
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(f"{fig_path}/true_{targets}_pred_{preds}.png")
    # plt.show()


def start_test(allowed_suffixes, model_type='test'):
    def test_model(model):

        correct_num = 0
        ft_all, lbl_all = None, None
        total_preds,total_targets = [], []
        with torch.no_grad():
            for i, (centers, corners, normals, neighbor_index, targets) in enumerate(data_loader):
                centers = centers.to(device)
                corners = corners.to(device)
                normals = normals.to(device)
                neighbor_index = neighbor_index.to(device)
                targets = targets.to(device)

                outputs, feas = model(centers, corners, normals, neighbor_index)
                _, preds = torch.max(outputs, 1)

                correct_num += (preds == targets).float().sum()

                if cfg['retrieval_on']:
                    ft_all = append_feature(ft_all, feas.detach().cpu())
                    lbl_all = append_feature(lbl_all, targets.detach().cpu(), flaten=True)
                # plot_mesh(centers=centers, corners=corners, normals=normals,
                #             targets=targets,
                #             preds=preds,
                #             model_type=model_type)
                total_preds.extend(preds.detach().cpu().numpy().tolist())
                total_targets.extend(targets.detach().cpu().numpy().tolist())
        print('\tAccuracy: {:.4f}'.format(float(correct_num) / len(data_set)))
        if cfg['retrieval_on']:
            print('\tmAP: {:.4f}'.format(calculate_map(ft_all, lbl_all)))
        return total_targets, total_preds


    cfg = get_test_config(model_type)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['cuda_devices']
    data_set = ModelNet10(cfg=cfg['dataset'], part='test')

    # if model_type != 'attack':
    #     data_set.get_filtered_dataset(allowed_suffixes) 
    data_set.get_filtered_dataset(allowed_suffixes)
    data_loader = data.DataLoader(data_set, batch_size=cfg['batch_size'], num_workers=4, shuffle=False, pin_memory=False)
    
    model = MeshNet(cfg=cfg['MeshNet'], require_fea=True)
    model.to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(cfg['load_model']))
    model.eval()
    test_targets , test_preds = test_model(model)

    paths_used = [data_set.files_names[i] for i in range(len(data_set))]
    return test_targets , test_preds, paths_used

    # return test_targets , test_preds

def eval_meshnet_attack(allowed_suffixes):
    true_labels_clean, clean_preds, clean_paths = start_test(model_type='mesh', allowed_suffixes=allowed_suffixes)
    true_labels_adv, adv_preds, adv_paths = start_test(model_type='attack', allowed_suffixes=allowed_suffixes)

    print(f"assert set(clean_paths) == set(adv_paths) --> {set(clean_paths) == set(adv_paths)}")

    clean_df = pd.DataFrame({
        'path': clean_paths,
        'true': true_labels_clean,
        'clean': clean_preds
    })
    adv_df = pd.DataFrame({
        'path': adv_paths,
        'true_adv': true_labels_adv,
        'adv': adv_preds
    })

    merged_df = clean_df.merge(adv_df[['path', 'adv']], on='path', how='inner')
    merged_df.to_csv(f"results/merged_meshnet{attack_params}.csv", index=False)

    print("Clean MeshNet Report:")
    print(classification_report(merged_df["true"], merged_df["clean"],target_names=class_names))

    print("Attacked MeshNet Report:")
    print(classification_report(merged_df["true"], merged_df["adv"],target_names=class_names))

def eval_pointnet_attack(csv_path):
    create_report(csv_path, model="PointNet++")


if __name__ == "__main__":
    model = "MeshNet"
    model = "PointNetPP"
    attack_params = "_pts_2048_lowpass_spec0125" # if model = "MeshNet" else "_lowpass100"
    attack_params = "_filtered_le2048_pts_2048_lowpass_spec0125"
    results_path = f"data/attack_results{attack_params}.csv"
    results_df = pd.read_csv(results_path)
    preprocess_df = True
    if preprocess_df:
        allowed_suffixes = results_df['clean_path'].apply(lambda x: x.split("../dataset/ModelNet10/")[-1].replace(".off",".npz"))
        # results_df['allowed_suffixes'] = allowed_suffixes.apply(lambda x: f"dataset/ModelNet10_processed/{x}")
        results_df['allowed_suffixes'] = allowed_suffixes
        results_df.to_csv(results_path, index=False)
    allowed_suffixes = set(results_df['allowed_suffixes'].to_list())
    eval_meshnet_attack(allowed_suffixes)
    eval_pointnet_attack(csv_path = results_path)
    # print("Clean MeshNet scores:")
    # test_targets , test_preds = start_test(model_type='mesh',allowed_suffixes=allowed_suffixes)
    # orig_test_results = pd.DataFrame(list(zip(test_targets, test_preds)), columns=['true_labels', 'test_preds'])
    # orig_test_results.to_csv(f"results/orig{attack_params}.csv", index=False)

    # print("Attacked MeshNet scores:")
    # test_targets2 , adv_test_preds = start_test(model_type = 'attack', allowed_suffixes=allowed_suffixes)
    # adv_test_results = pd.DataFrame(list(zip(test_targets2, adv_test_preds)), columns=['true_labels', 'adv_test_preds'])
    # adv_test_results.to_csv(f"results/adv{attack_params}.csv", index=False)
    
    # cm_path = f"results/cm{attack_params}"
    # plot_confusion_matrix_comparison(csv_path=results_path, save_to_path=cm_path,model="PointNet++")
    # create_report(csv_path=results_path,model="PointNet++")
    
    # create_point_net_reports = False
    # if create_point_net_reports:
    #     cm_path = f"results/cm_pointnetpp_lowpass100"
    #     plot_confusion_matrix_comparison(csv_path=f"data/pointnetpp_attack_results_lowpass100.csv", save_to_path=cm_path,model="PointNet++")
    #     create_report(csv_path=f"data/pointnetpp_attack_results_lowpass100.csv",model="PointNet++")
        