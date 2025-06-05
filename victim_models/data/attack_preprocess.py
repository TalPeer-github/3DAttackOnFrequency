import pymeshlab
import numpy as np
from pathlib import Path
from rich.progress import track
# from config.config import modelnet10_labels, filtered_classes, filtered_labels

modelnet10_labels = {
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

modelnet10_classes = {
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

filtered_classes = {
    1:'bed',
    2:'chair',
    7:'sofa',
    9:'toilet',
}

filtered_labels = {
    'bed': 1,
    'chair': 2,
    'sofa': 7,
    'toilet': 9,
}

def find_neighbor_old(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face

def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            return i
    return except_face

def init_attacker():
    model = PointNetPlusPlus(
        args.set_abstraction_ratio_1,
        args.set_abstraction_ratio_2,
        args.set_abstraction_radius_1,
        args.set_abstraction_radius_2,
        args.dropout
    ).to(args.device)

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)

    return CWAOF(
        model=model,
        device=args.device,
        eps=args.epsilon,
        steps=args.num_steps,
        lr=args.lr,
        budget=args.epsilon,
        gamma=args.gamma,
        max_iter=args.max_iter
    )

import os
import numpy as np
import torch
import trimesh
import pymeshlab
from torch_geometric.data import Data
from pathlib import Path
from rich.progress import track

from point_cloud_attack import CWAOF
from victim_models.point_net_pp import PointNetPlusPlus
from utils.config import args, modelnet10_labels, modelnet10_classes


def init_attacker():
    model = PointNetPlusPlus(
        args.set_abstraction_ratio_1,
        args.set_abstraction_ratio_2,
        args.set_abstraction_radius_1,
        args.set_abstraction_radius_2,
        args.dropout
    ).to(args.device)

    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)

    return CWAOF(
        model=model,
        device=args.device,
        eps=args.epsilon,
        steps=args.num_steps,
        lr=args.lr,
        budget=args.epsilon,
        gamma=args.gamma,
        max_iter=args.max_iter
    )


if __name__ == '__main__':
    root = Path('../dataset/ModelNet10')
    new_root = Path('../dataset/ModelNet10_attacked')
    max_faces = 1000
    ms = pymeshlab.MeshSet()
    shape_list = sorted(list(root.glob('*/*/*.off')))
    attacker = init_attacker()

    print(f"Found {len(shape_list)} .off files.")
    total_skipped = 0

    for shape_path in track(shape_list):
        class_name = shape_path.parts[-3]
        label = modelnet10_labels.get(class_name, -1)
        if label == -1:
            continue

        ms.clear()
        ms.load_new_mesh(str(shape_path))
        mesh = ms.current_mesh()
        vertices = mesh.vertex_matrix()
        faces = mesh.face_matrix()
        y = torch.tensor(label, dtype=torch.long)

        if faces.shape[0] > max_faces:
            print(f"Skipping (>{max_faces} faces): {shape_path}")
            total_skipped += 1
            continue

        center = (np.max(vertices, axis=0) + np.min(vertices, axis=0)) / 2
        vertices -= center
        max_len = np.max(np.sum(vertices**2, axis=1))
        vertices /= np.sqrt(max_len)

        pc_data = Data(pos=torch.tensor(vertices, dtype=torch.float32), face=torch.tensor(faces.T, dtype=torch.long), y=y)
        adv_vertices, _, adv_prediction = attacker.attack(pc_data)

        adv_vertices_np = adv_vertices.cpu().numpy()
        adv_mesh = trimesh.Trimesh(vertices=adv_vertices_np, faces=faces, process=False)

        rel_path = shape_path.relative_to(root)
        out_path = new_root / rel_path.with_suffix(f'_adv{adv_prediction.item()}.off')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        adv_mesh.export(out_path)

    print(f"✅ Finished. Skipped {total_skipped} files due to face count.")