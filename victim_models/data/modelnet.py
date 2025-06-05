import numpy as np
import os
from pathlib import Path
import torch
import torch.utils.data as data
import pymeshlab
from data.preprocess import find_neighbor
from config.config import modelnet10_labels, filtered_classes, filtered_labels



class ModelNet10(data.Dataset):

    def __init__(self, cfg, part='train',use_filtered_classes=False,):
        """
        cfg: A configuration dictionary, typically passed from a YAML or argument parser.
        part: "train" or "test", specifying which split to load.
        """
        self.root = cfg['data_root']            # Root path to dataset
        self.max_faces = cfg['max_faces']       # Max faces to include per mesh
        self.part = part                        # train/test split
        self.augment_data = cfg['augment_data'] # Whether to apply data augmentation
        self.labels = filtered_labels if use_filtered_classes else modelnet10_labels
        self.data = []

        for lbl in os.listdir(self.root):
            if lbl not in self.labels:
                continue
            lbl_idx = self.labels[lbl]
            lbl_dir = os.path.join(self.root, lbl, part)
            for fname in os.listdir(lbl_dir):
                if fname.endswith('.npz') or fname.endswith('.off'):
                    self.data.append((os.path.join(lbl_dir, fname), lbl_idx))


    def get_filtered_dataset(self,allowed_suffixes):
        """
        filter test samples that was attacked
        """
        filtered_data = []
        files_names = []
        for i in range(len(self.data)):
            path, type = self.data[i]
            suff = "/".join(Path(path).parts[-3:])
            if str(suff) in allowed_suffixes:
                filtered_data.append(self.data[i])
                files_names.append(suff)
            else:
                print("===========================================")
                print(f"Suff: \t{suff} \nnot in allowed suffixes.")
                print("===========================================")
        self.data = filtered_data
        self.files_names = files_names
        
                

    def __getitem__(self, i):
        path, type = self.data[i]
        if path.endswith('.npz'):
            data = np.load(path)
            face = data['faces']
            neighbor_index = data['neighbors']
        else:
            face, neighbor_index = process_mesh(path, self.max_faces)
            
            if face is None:
                print("FACE is None ! (data/modelnet.py -> __getitem__()")
                return self.__getitem__(0)
            
        num_points = len(face)
        if num_points < self.max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(self.max_faces - num_points):
                index = np.random.randint(0, num_points)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))
        
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()
        target = torch.tensor(type, dtype=torch.long)

        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        return centers, corners, normals, neighbor_index, target

    def __len__(self):
        return len(self.data)
    

def process_mesh(path, max_faces):
    ms = pymeshlab.MeshSet()
    ms.clear()

    ms.load_new_mesh(path)
    mesh = ms.current_mesh()
    
    # # clean up
    # mesh, _ = pymesh.remove_isolated_vertices(mesh)
    # mesh, _ = pymesh.remove_duplicated_vertices(mesh)

    # get elements
    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()

    if faces.shape[0] > max_faces:   
        print("Model with more than {} faces ({}): {}".format(max_faces, faces.shape[0], path))
        return None, None

    center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
    vertices -= center


    max_len = np.max(vertices[:, 0]**2 + vertices[:, 1]**2 + vertices[:, 2]**2)
    vertices /= np.sqrt(max_len)


    ms.clear()
    mesh = pymeshlab.Mesh(vertices, faces)
    ms.add_mesh(mesh)
    face_normal = ms.current_mesh().face_normal_matrix()

    faces_contain_this_vertex = []
    for i in range(len(vertices)):
        faces_contain_this_vertex.append(set([]))
    centers = []
    corners = []
    for i in range(len(faces)):
        [v1, v2, v3] = faces[i]
        x1, y1, z1 = vertices[v1]
        x2, y2, z2 = vertices[v2]
        x3, y3, z3 = vertices[v3]
        centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
        corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
        faces_contain_this_vertex[v1].add(i)
        faces_contain_this_vertex[v2].add(i)
        faces_contain_this_vertex[v3].add(i)

    neighbors = []
    for i in range(len(faces)):
        [v1, v2, v3] = faces[i]
        n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
        n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
        n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
        neighbors.append([n1, n2, n3])

    centers = np.array(centers)
    corners = np.array(corners)
    faces = np.concatenate([centers, corners, face_normal], axis=1)
    neighbors = np.array(neighbors)

    return faces, neighbors