import os
from collections import defaultdict
import csv
from pathlib import Path, PosixPath
from rich.progress import track

import numpy as np

import trimesh
import pymeshlab

import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch.backends.cudnn as cudnn

from point_cloud_attack import CWAOF
from victim_models.point_net_pp import PointNetPlusPlus
from utils.config import args, modelnet10_labels, modelnet10_classes, label_to_class, class_to_label
from utils.visalizations import plot_mesh_from_arrays, plot_mesh_comparison

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        low_pass=low_pass,
    )


def get_shape_list(root='victim_models/dataset/ModelNet10', max_faces=1000, max_points=2048):
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


def init_model(filtered_train=True):
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
        batch=torch.zeros(vertices.shape[0], dtype=torch.long),
    )

    with torch.no_grad():
        pc_data = pc_data.to(device)
        clean_logits = pc_model(pc_data)
        clean_prediction = torch.argmax(clean_logits, dim=-1)

    clean_prediction = clean_prediction.cpu().item()
    new_ms.clear()
    return clean_prediction


def create_results_df(attack_params, max_faces=1000, max_points=2048, low_pass=50):
    root = Path(f'victim_models/dataset/ModelNet10_attacked{attack_params}')
    shape_list = get_shape_list(root, max_faces, max_points)

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
            batch=torch.zeros(vertices.shape[0], dtype=torch.long),
        )

        print(f"Evaluating attack ({class_name}):\t {pc_data}")
        with torch.no_grad():
            pc_data = pc_data.to(device)
            adv_logits = pc_model(pc_data)
            adv_prediction = torch.argmax(adv_logits, dim=-1)
        adv_prediction = adv_prediction.cpu().item()
        clean_prediction = get_clean_prediction(shape_path, pc_model)

        print(f"Mesh Class = {label}")
        print(f"\t Clean Prediction Class = {clean_prediction}\n\n")
        print(f"\t Adversarial Class = {adv_prediction}\n\n")

        log_file.write(f"{shape_path} | true: {label} | clean: {clean_prediction} | adv: {adv_prediction}\n")

    log_file.close()

    print(f"Finished.")


def results_to_csv(attack_params, lowpass=0.1, specific_lowpass=False, threshold=0.1,
                   dir_path="victim_models/data/results"):
    input_path = f"attack_log{attack_params}.txt"
    output_path = f"{dir_path}/attack_results{attack_params}.csv"

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


def start_attack(attack_params, max_faces=1000, max_points=2048, low_pass=75, specific_lowpass=False, threshold=0.1,
                 filtered_train=True):
    root = Path('victim_models/dataset/ModelNet10')
    new_root = Path(f'victim_models/dataset/ModelNet10_attacked{attack_params}')

    shape_list = get_shape_list(root, max_faces, max_points)

    ms = pymeshlab.MeshSet()
    pc_attacker = init_attacker(low_pass=low_pass, filtered_train=filtered_train)

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
        max_len = np.max(np.sum(vertices ** 2, axis=1))
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
        adv_vertices, clean_prediction, adv_prediction = pc_attacker.attack(pc_data, specific_lowpass=specific_lowpass)
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
        log_file.write(
            f"{shape_path} | true: {label} | clean: {clean_prediction} | adv: {adv_prediction} | lowpass: {int(threshold * V)}\n")

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
    start_attack(low_pass=low_pass, max_points=max_points, specific_lowpass=specific_lowpass, threshold=threshold,
                 attack_params=attack_params, filtered_train=filtered_le2048)
    results_to_csv(attack_params=attack_params, specific_lowpass=True)
