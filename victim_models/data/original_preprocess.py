import pymeshlab
import numpy as np
from pathlib import Path
from rich.progress import track
from collections import defaultdict

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

modelnet10_labels = label_to_class
modelnet10_classes = class_to_label


def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face

def get_shape_list(root = '../dataset/ModelNet10',max_points=2048):
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
        
        new_path = Path('/'.join(path.parts[-5:]))
        # class_to_files[class_name].append(new_path)

        # path = new_path
        class_to_files[class_name].append(path)

    shape_list = []
    for _, files in class_to_files.items():
        shape_list.extend(files[1::2]) 

    print(f"Found {len(shape_list)} .off files.")
    return shape_list

if __name__ == '__main__':
    root = Path('../dataset/ModelNet10')
    new_root = Path('../dataset/ModelNet10_filtered_processed')
    max_faces = 20000
    shape_list = get_shape_list()
    ms = pymeshlab.MeshSet()

    for shape_dir in track(shape_list):
        out_dir = new_root / shape_dir.relative_to(root).with_suffix('.npz')
        out_dir.parent.mkdir(parents=True, exist_ok=True)

        ms.clear()

        ms.load_new_mesh(str(shape_dir))
        mesh = ms.current_mesh()

        vertices = mesh.vertex_matrix()
        faces = mesh.face_matrix()

        if faces.shape[0] >= max_faces:
            print("Model with more than {} faces ({}): {}".format(max_faces, faces.shape[0], out_dir))
            continue

 
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

        np.savez(str(out_dir), faces=faces, neighbors=neighbors)