import os
import csv
from pathlib import Path, PosixPath
from rich.progress import track
from collections import defaultdict

import numpy as np
import torch
import trimesh
import pymeshlab
import torch.nn as nn
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from point_cloud_attack import CWAOF
from victim_models.point_net_pp import PointNetPlusPlus
from utils.config import args, modelnet10_labels, modelnet10_classes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_70_shape_list_print = [PosixPath('victim_models/dataset/ModelNet10/bathtub/test/bathtub_0108.off'), PosixPath('victim_models/dataset/ModelNet10/bathtub/test/bathtub_0116.off'), PosixPath('victim_models/dataset/ModelNet10/bathtub/test/bathtub_0120.off'), PosixPath('victim_models/dataset/ModelNet10/bathtub/test/bathtub_0126.off'), PosixPath('victim_models/dataset/ModelNet10/bathtub/test/bathtub_0127.off'), PosixPath('victim_models/dataset/ModelNet10/bed/test/bed_0519.off'), PosixPath('victim_models/dataset/ModelNet10/bed/test/bed_0520.off'), PosixPath('victim_models/dataset/ModelNet10/bed/test/bed_0523.off'), PosixPath('victim_models/dataset/ModelNet10/bed/test/bed_0526.off'), PosixPath('victim_models/dataset/ModelNet10/bed/test/bed_0529.off'), PosixPath('victim_models/dataset/ModelNet10/chair/test/chair_0907.off'), PosixPath('victim_models/dataset/ModelNet10/chair/test/chair_0908.off'), PosixPath('victim_models/dataset/ModelNet10/chair/test/chair_0913.off'), PosixPath('victim_models/dataset/ModelNet10/chair/test/chair_0925.off'), PosixPath('victim_models/dataset/ModelNet10/chair/test/chair_0939.off'), PosixPath('victim_models/dataset/ModelNet10/desk/test/desk_0202.off'), PosixPath('victim_models/dataset/ModelNet10/desk/test/desk_0203.off'), PosixPath('victim_models/dataset/ModelNet10/desk/test/desk_0205.off'), PosixPath('victim_models/dataset/ModelNet10/desk/test/desk_0206.off'), PosixPath('victim_models/dataset/ModelNet10/desk/test/desk_0207.off'), PosixPath('victim_models/dataset/ModelNet10/dresser/test/dresser_0201.off'), PosixPath('victim_models/dataset/ModelNet10/dresser/test/dresser_0203.off'), PosixPath('victim_models/dataset/ModelNet10/dresser/test/dresser_0205.off'), PosixPath('victim_models/dataset/ModelNet10/dresser/test/dresser_0206.off'), PosixPath('victim_models/dataset/ModelNet10/dresser/test/dresser_0209.off'), PosixPath('victim_models/dataset/ModelNet10/monitor/test/monitor_0466.off'), PosixPath('victim_models/dataset/ModelNet10/monitor/test/monitor_0468.off'), PosixPath('victim_models/dataset/ModelNet10/monitor/test/monitor_0475.off'), PosixPath('victim_models/dataset/ModelNet10/monitor/test/monitor_0478.off'), PosixPath('victim_models/dataset/ModelNet10/monitor/test/monitor_0480.off'), PosixPath('victim_models/dataset/ModelNet10/night_stand/test/night_stand_0202.off'), PosixPath('victim_models/dataset/ModelNet10/night_stand/test/night_stand_0207.off'), PosixPath('victim_models/dataset/ModelNet10/night_stand/test/night_stand_0209.off'), PosixPath('victim_models/dataset/ModelNet10/night_stand/test/night_stand_0210.off'), PosixPath('victim_models/dataset/ModelNet10/night_stand/test/night_stand_0212.off'), PosixPath('victim_models/dataset/ModelNet10/sofa/test/sofa_0681.off'), PosixPath('victim_models/dataset/ModelNet10/sofa/test/sofa_0686.off'), PosixPath('victim_models/dataset/ModelNet10/sofa/test/sofa_0687.off'), PosixPath('victim_models/dataset/ModelNet10/sofa/test/sofa_0693.off'), PosixPath('victim_models/dataset/ModelNet10/sofa/test/sofa_0695.off'), PosixPath('victim_models/dataset/ModelNet10/table/test/table_0394.off'), PosixPath('victim_models/dataset/ModelNet10/table/test/table_0395.off'), PosixPath('victim_models/dataset/ModelNet10/table/test/table_0397.off'), PosixPath('victim_models/dataset/ModelNet10/table/test/table_0398.off'), PosixPath('victim_models/dataset/ModelNet10/table/test/table_0401.off'), PosixPath('victim_models/dataset/ModelNet10/toilet/test/toilet_0348.off'), PosixPath('victim_models/dataset/ModelNet10/toilet/test/toilet_0353.off'), PosixPath('victim_models/dataset/ModelNet10/toilet/test/toilet_0354.off'), PosixPath('victim_models/dataset/ModelNet10/toilet/test/toilet_0363.off'), PosixPath('victim_models/dataset/ModelNet10/toilet/test/toilet_0390.off')]
_30_shape_list_print = [PosixPath('victim_models/dataset/ModelNet10/bathtub/test/bathtub_0120.off'), PosixPath('victim_models/dataset/ModelNet10/bathtub/test/bathtub_0126.off'), PosixPath('victim_models/dataset/ModelNet10/bathtub/test/bathtub_0127.off'), PosixPath('victim_models/dataset/ModelNet10/bed/test/bed_0523.off'), PosixPath('victim_models/dataset/ModelNet10/bed/test/bed_0526.off'), PosixPath('victim_models/dataset/ModelNet10/bed/test/bed_0529.off'), PosixPath('victim_models/dataset/ModelNet10/chair/test/chair_0913.off'), PosixPath('victim_models/dataset/ModelNet10/chair/test/chair_0925.off'), PosixPath('victim_models/dataset/ModelNet10/chair/test/chair_0939.off'), PosixPath('victim_models/dataset/ModelNet10/desk/test/desk_0205.off'), PosixPath('victim_models/dataset/ModelNet10/desk/test/desk_0206.off'), PosixPath('victim_models/dataset/ModelNet10/desk/test/desk_0207.off'), PosixPath('victim_models/dataset/ModelNet10/dresser/test/dresser_0205.off'), PosixPath('victim_models/dataset/ModelNet10/dresser/test/dresser_0206.off'), PosixPath('victim_models/dataset/ModelNet10/dresser/test/dresser_0209.off'), PosixPath('victim_models/dataset/ModelNet10/monitor/test/monitor_0475.off'), PosixPath('victim_models/dataset/ModelNet10/monitor/test/monitor_0478.off'), PosixPath('victim_models/dataset/ModelNet10/monitor/test/monitor_0480.off'), PosixPath('victim_models/dataset/ModelNet10/night_stand/test/night_stand_0209.off'), PosixPath('victim_models/dataset/ModelNet10/night_stand/test/night_stand_0210.off'), PosixPath('victim_models/dataset/ModelNet10/night_stand/test/night_stand_0212.off'), PosixPath('victim_models/dataset/ModelNet10/sofa/test/sofa_0687.off'), PosixPath('victim_models/dataset/ModelNet10/sofa/test/sofa_0693.off'), PosixPath('victim_models/dataset/ModelNet10/sofa/test/sofa_0695.off'), PosixPath('victim_models/dataset/ModelNet10/table/test/table_0397.off'), PosixPath('victim_models/dataset/ModelNet10/table/test/table_0398.off'), PosixPath('victim_models/dataset/ModelNet10/table/test/table_0401.off'), PosixPath('victim_models/dataset/ModelNet10/toilet/test/toilet_0354.off'), PosixPath('victim_models/dataset/ModelNet10/toilet/test/toilet_0363.off'), PosixPath('victim_models/dataset/ModelNet10/toilet/test/toilet_0390.off')]

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



def plot_mesh_from_arrays(vertices, faces, show=False, save_path=None):
    faces = faces.astype(int)  # ensure indices are integers
    corners = vertices[faces]  # shape: (F, 3, 3)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(corners,alpha=0.25, edgecolor='#670D2F',facecolor='#EF88AD')
    ax.add_collection3d(mesh)
    
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                edgecolor='#670D2F', s=50, alpha=0.3,c='#EF88AD',linewidth=1.25)

    max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max() / 2
    mid = vertices.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_box_aspect([1, 1, 1])


    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}.png", bbox_inches='tight')

    if show:
        plt.show()
    plt.close()

   
    
    
def plot_mesh_comparison(clean_v, adv_v, faces, label, clean_pred, adv_pred, save_path=None):
    """
    Visualizes clean vs. adversarial mesh side-by-side.
    """
    faces = faces.astype(int)
    corners_clean = clean_v[faces]  # (F, 3, 3)
    corners_adv = adv_v[faces]
    true_class = class_to_label[label].capitalize()
    pred_class = class_to_label[clean_pred].capitalize()
    adv_class = class_to_label[adv_pred].capitalize()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), subplot_kw={'projection': '3d'},)
    mesh1 = Poly3DCollection(corners_clean,alpha=0.25, facecolor='lightblue', edgecolor='k')
    axes[0].add_collection3d(mesh1)
    axes[0].set_title(f"Clean Prediction: {pred_class}")
    axes[0].set_box_aspect([1, 1, 1])
    axes[0].set_axis_off()
    

    mesh2 = Poly3DCollection(corners_adv,alpha=0.25, edgecolor='#670D2F',facecolor='#EF88AD')
    axes[1].add_collection3d(mesh2)
    axes[1].set_title(f"Attack Prediction: {adv_class}")
    axes[1].set_axis_off()
    axes[1].set_box_aspect([1, 1, 1])
    axes[1].set_axis_off()

    for ax, verts in zip([axes[0], axes[1]], [clean_v, adv_v]):
        max_range = (verts.max(axis=0) - verts.min(axis=0)).max() / 2
        mid = verts.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    plt.suptitle(f"Adversarial Mesh (True label = {true_class})",fontsize=16,fontweight='bold')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def init_attacker(low_pass=100, filtered_train=True):
    model = init_model(filtered_train)

    return CWAOF(
        model=model,
        device=args.device,
        eps=args.epsilon,
        steps=args.num_steps,
        lr=args.lr,
        budget=args.epsilon,
        gamma=args.gamma,
        max_iter=args.max_iter,
        low_pass = low_pass,
    )

def get_shape_list(root = 'victim_models/dataset/ModelNet10', max_faces=1000,max_points=2048):
    root = Path(root)
    ms = pymeshlab.MeshSet()
    test_files = sorted(list(root.glob('*/*/*.off')))
    class_to_files = defaultdict(list)
    
    total_skipped = 0
    for path in test_files:
        if path.parts[-2] != 'test':
            continue
        class_name = path.parts[-3]
        label = modelnet10_labels.get(class_name, -1)

        ms.clear()
        ms.load_new_mesh(str(path))
        mesh = ms.current_mesh()
        vertices = mesh.vertex_matrix()

        if vertices.shape[0] > max_points:
            print(f"Skipping (>{max_points} points): {path}")
            total_skipped += 1
            continue
        
        class_to_files[class_name].append(path)

    shape_list = []
    for _, files in class_to_files.items():
        shape_list.extend(files[1::2]) 

    print(f"Found {len(shape_list)} .off files.")


    return shape_list

def init_model(filtered_train = True):
    model = PointNetPlusPlus(
        set_abstraction_ratio_1=args.set_abstraction_ratio_1,
        set_abstraction_ratio_2=args.set_abstraction_ratio_2,
        set_abstraction_radius_1=args.set_abstraction_radius_1,
        set_abstraction_radius_2=args.set_abstraction_radius_2,
        dropout=args.dropout).to(args.device)
    if filtered_train:
        checkpoint = torch.load("victim_models/checkpoints/filtered_le2048_PointNetPP_epochs19_val_acc81.pt")
    else:
        checkpoint = torch.load("victim_models/checkpoints/PointNetPP_epochs10_val_acc90.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def get_clean_prediction(shape_path, pc_model):
    new_ms = pymeshlab.MeshSet()
    class_name = shape_path.parts[-3]
    clean_path = Path(f'victim_models/dataset/ModelNet10/{class_name}/test/{shape_path.parts[-1]}')
    label = label_to_class[class_name]
    new_ms.load_new_mesh(str(shape_path))
    mesh = new_ms.current_mesh()
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()
    y = torch.tensor([label], dtype=torch.long)
    V = vertices.shape[0]
    pc_data = Data(
        pos=torch.tensor(vertices, dtype=torch.float),
        face=torch.tensor(faces.T, dtype=torch.long),
        y=y,
        batch = torch.zeros(vertices.shape[0], dtype=torch.long),
    )

    with torch.no_grad():
        pc_data = pc_data.to(device)
        clean_logits = pc_model(pc_data)
        clean_prediction = torch.argmax(clean_logits, dim=-1)

    clean_prediction = clean_prediction.cpu().item()
    new_ms.clear()
    return clean_prediction

def create_results_df(attack_params, max_faces=1000,max_points=2048, low_pass=50):
    root = Path(f'victim_models/dataset/ModelNet10_attacked{attack_params}')
    shape_list = get_shape_list(root,max_faces,max_points)

    ms = pymeshlab.MeshSet()
    pc_model = init_model()

    log_file = open(f"attack_log{attack_params}.txt", "w")
    log_file.write(f"path,true,clean,adv\n")

    for shape_path in track(shape_list):
        class_name = shape_path.parts[-3]
        label = label_to_class[class_name]

        ms.clear()
        ms.load_new_mesh(str(shape_path))
        mesh = ms.current_mesh()
        vertices = mesh.vertex_matrix()
        faces = mesh.face_matrix()
        y = torch.tensor([label], dtype=torch.long)

        V = vertices.shape[0]
        pc_data = Data(
            pos=torch.tensor(vertices, dtype=torch.float),
            face=torch.tensor(faces.T, dtype=torch.long),
            y=y,
            batch = torch.zeros(vertices.shape[0], dtype=torch.long),
        )
 
        print(f"Evaluating attack ({class_name}):\t {pc_data}")
        with torch.no_grad():
            pc_data = pc_data.to(device)
            adv_logits = pc_model(pc_data)
            adv_prediction = torch.argmax(adv_logits, dim=-1)
        adv_prediction = adv_prediction.cpu().item()
        clean_prediction = get_clean_prediction(shape_path,pc_model)

        print(f"Mesh Class = {label}")
        print(f"\t Clean Prediction Class = {clean_prediction}\n\n")
        print(f"\t Adversarial Class = {adv_prediction}\n\n")

        log_file.write(f"{shape_path} | true: {label} | clean: {clean_prediction} | adv: {adv_prediction}\n")

        
    log_file.close()    

    print(f"Finished.")

def results_to_csv(attack_params,lowpass=0.1,specific_lowpass=False,threshold=0.1):
    input_path = f"attack_log{attack_params}.txt"
    output_path = f"attack_results{attack_params}.csv"

    with open(input_path, "r") as log, open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["path", "true", "clean", "adv", "lowpass"])  # CSV header

        for line in log:
            parts = line.strip().split("|")
            if len(parts) != 5:
                continue  # skip malformed lines

            path = parts[0].strip()
            true_label = parts[1].split(":")[1].strip()
            clean_label = parts[2].split(":")[1].strip()
            adv_label = parts[3].split(":")[1].strip()
            lowpass = parts[4].split(":")[1].strip()
            
            writer.writerow([path, true_label, clean_label, adv_label, lowpass])

    print(f"CSV file saved to: {output_path}")

def start_attack(attack_params, max_faces=1000,max_points=2048, low_pass=75,specific_lowpass=False,threshold=0.1,filtered_train=True):
    root = Path('victim_models/dataset/ModelNet10')
    new_root = Path(f'victim_models/dataset/ModelNet10_attacked{attack_params}')

    shape_list = get_shape_list(root,max_faces,max_points)

    ms = pymeshlab.MeshSet()
    pc_attacker = init_attacker(low_pass = low_pass, filtered_train = filtered_train)

    log_file = open(f"attack_log{attack_params}.txt", "w")
    log_file.write(f"path,true,clean,adv,lowpass\n")

    for shape_path in track(shape_list):
        class_name = shape_path.parts[-3]
        label = label_to_class[class_name]

        ms.clear()
        ms.load_new_mesh(str(shape_path))
        mesh = ms.current_mesh()

        vertices = mesh.vertex_matrix()
        faces = mesh.face_matrix()
        y = torch.tensor([label], dtype=torch.long)

        center = (np.max(vertices, axis=0) + np.min(vertices, axis=0)) / 2
        vertices -= center
        max_len = np.max(np.sum(vertices**2, axis=1))
        vertices /= np.sqrt(max_len)

        V = vertices.shape[0]
        pc_data = Data(
            pos=torch.tensor(vertices, dtype=torch.float),
            face=torch.tensor(faces.T, dtype=torch.long),
            y=y,
        )

        pc_data.batch = torch.zeros(pc_data.pos.shape[0], dtype=torch.long)
        if specific_lowpass:
            pc_attacker.low_pass = int(threshold * V)
        print(f"Initializing attack ({class_name}):\t {pc_data}")
        adv_vertices, clean_prediction, adv_prediction = pc_attacker.attack(pc_data, specific_lowpass = specific_lowpass)
        clean_prediction = clean_prediction.cpu().item()
        adv_prediction = adv_prediction.cpu().item()

        print(f"Mesh Class = {label}")
        print(f"\t PointNet++ Class = {clean_prediction}")
        print(f"\t Adversarial Class = {adv_prediction}\n\n")

        adv_vertices_np = adv_vertices.cpu().numpy()
        original_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        adv_mesh = trimesh.Trimesh(vertices=adv_vertices_np, faces=faces, process=False)

        out_path = new_root / shape_path.relative_to(root).with_suffix('.off')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        adv_mesh.export(out_path)
        log_file.write(f"{shape_path} | true: {label} | clean: {clean_prediction} | adv: {adv_prediction} | lowpass: {int(threshold * V)}\n")

    log_file.close()    
    print(f"Finished (log path -> attack_log{attack_params}.txt).")

if __name__ == '__main__':
    low_pass = 50
    threshold = 0.125
    max_points = 2048
    specific_lowpass = True
    filtered_le2048 = True
    if specific_lowpass:
        spec_lp = f"{str(threshold).replace('.', '')}"
        attack_params = f"_pts_{max_points}_lowpass_spec{spec_lp}"
    else:
        attack_params = f"_pts_{max_points}_lowpass{low_pass}"
    if filtered_le2048:
        _attack_params = f"_filtered_le2048" + attack_params
        attack_params = _attack_params
    start_attack(low_pass=low_pass, max_points = max_points,specific_lowpass=specific_lowpass,threshold=threshold,attack_params=attack_params, filtered_train=filtered_le2048)
    # create_results_df(attack_params=attack_params)
    results_to_csv(attack_params=attack_params,specific_lowpass=True)
