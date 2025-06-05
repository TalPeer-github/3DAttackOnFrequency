import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import trimesh
import pymeshlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection




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



def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face

def plot_mesh_from_arrays(results_df,vertices,faces,class_name, shape_path,model_type='attack'):
    file = shape_path.parts[-1]
    file_mask = results_df['file']==file
    true = modelnet10_classes[results_df.loc[file_mask ,"true"].item()]
    pnetpp_pred = modelnet10_classes[results_df.loc[file_mask ,"clean"].item()]
    attack_pred = modelnet10_classes[results_df.loc[file_mask ,"adv"].item()]

    print(f"assert true == class_name --> {true == class_name}")

    faces = faces.astype(int)
    corners = vertices[faces]  

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(corners, alpha=0.7, edgecolor='k', linewidth=0.3)
    ax.add_collection3d(mesh)

    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='red', s=8, alpha=0.6, label='Vertices')

    max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max() / 2
    mid = vertices.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    if model_type == 'original':
        fig_title = f"Attacked Point Cloud to Mesh \nTrue label = {class_name}"
    else:
        fig_title = f"Attacked Point Cloud to Mesh \nTrue label = {class_name} | PointNet++ Prediction = {pnetpp_pred}\n({model_type}) | Attack Prediction = {adv_pred}"
    ax.set_title(fig_title)
    
    plt.tight_layout()
    fig_path = f"../../attacks/visualizations/{model_type}"
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(f"{fig_path}/true_{class_name}_pred_{pointnetpp_pred}.png")
    plt.close()



def get_original_mesh(file, max_faces = 1000,max_points=2048,filter_by_faces = False,filter_by_points=False):
    """
    shape_list item : PosixPath('../dataset/ModelNet10/bathtub/test/bathtub_0126.off')
    file = ../dataset/ModelNet10_attacked/bathtub/test/bathtub_0126.off
    """
    ms = pymeshlab.MeshSet()
    ms.clear()
    ms.load_new_mesh(file)
    mesh = ms.current_mesh()

    vertices = mesh.vertex_matrix()
    faces = mesh.face_matrix()

    if filter_by_faces:
        if faces.shape[0] > max_faces:
            print(f"Skipping (>{max_faces} faces): {file}")
            total_skiped += 1
            return

    if filter_by_points:
        if vertices.shape[0] > max_points:
            print(f"Skipping (>{max_points} points): {file}")
            total_skiped += 1
            return

    center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
    vertices -= center
    max_len = np.max(np.sum(vertices**2, axis=1))
    vertices /= np.sqrt(max_len)
    
    original_vertices = vertices
    original_faces = faces

    # ms.clear()
    
    return original_vertices, original_faces


def plot_mesh_comparison(results_df, shape_path, clean_v, adv_v, faces, save_to_path="visualizations/comparisons/"):
    """
    Visualizes clean vs. adversarial mesh side-by-side.
    """

    file = shape_path.split('/')[-1].split('.')[0]
    save_path= save_to_path + f"{file}.png"
    
    file_mask = results_df['adv_path'] == shape_path
    true_class = modelnet10_classes[results_df.loc[file_mask ,"true"].values[0]]
    pred_class = modelnet10_classes[results_df.loc[file_mask ,"clean"].values[0]]
    adv_class = modelnet10_classes[results_df.loc[file_mask ,"adv"].values[0]]

    faces = faces.astype(int)
    corners_clean = clean_v[faces] 
    corners_adv = adv_v[faces]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 7), subplot_kw={'projection': '3d'})
    
    
    mesh1 = Poly3DCollection(corners_clean,alpha=0.2, facecolor='#FFEDFA', edgecolor='#123458',linewidth=1.5)
    axes[0].add_collection3d(mesh1)
    axes[0].set_title(f"Clean Prediction: {pred_class}")
   
    
    mesh2 = Poly3DCollection(corners_adv,alpha=0.2, facecolor='#FFEDFA', edgecolor='#123458',linewidth=1.5)
    axes[1].add_collection3d(mesh2)
    axes[1].set_title(f"Attack Prediction: {adv_class}")


    for ax, verts in zip([axes[0], axes[1]], [clean_v, adv_v]):
        max_range = (verts.max(axis=0) - verts.min(axis=0)).max() / 2
        mid = verts.mean(axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()

    plt.suptitle(f"Adversarial Mesh \n(True label = {true_class})", fontsize=14, fontweight='bold', y=0.9)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()



def process_attack_to_mesh(attack_params,max_faces = 1000,max_points=2048,filter_by_faces = False,filter_by_points=True,use_full=True):
    root = Path(f'../dataset/ModelNet10_attacked{attack_params}')
    new_root = Path(f'../dataset/ModelNet10_attacked_processed{attack_params}_success') if not use_full else Path(f'../dataset/ModelNet10_attacked{attack_params}_processed')

    attack_results_df = attack_results[attack_results['adv_success'] == True] if not use_full else attack_results
    
    original_shape_list = attack_results_df['clean_path'].to_list()
    attacked_shape_list = attack_results_df['adv_path'].to_list()
    original_classes = attack_results_df['true'].to_list()

    
    total_skiped = 0
    ms = pymeshlab.MeshSet()
    for original_shape_file,attacked_shape_file, class_name in zip(original_shape_list,attacked_shape_list,original_classes):

        ms.clear()
        ms.load_new_mesh(attacked_shape_file)
        mesh = ms.current_mesh()

        adv_vertices = mesh.vertex_matrix()
        adv_faces = mesh.face_matrix()

        if filter_by_faces:
            if adv_faces.shape[0] > max_faces:
                print(f"Skipping (>{max_faces} faces): {attacked_shape_file}")
                total_skiped += 1
                continue
        if filter_by_points:
            if adv_vertices.shape[0] > max_points:
                print(f"Skipping (>{max_points} points): {attacked_shape_file}")
                total_skiped += 1
                continue
        
        ms.clear()
        try:
            original_vertices, original_faces = get_original_mesh(file = original_shape_file,max_faces=max_faces,max_points=max_points,filter_by_points=filter_by_points,filter_by_faces=filter_by_faces,)
        except Exception as e:
            print(f"Problem with {original_shape_file} (get_original_mesh)")
            continue
        
        #save_to_path= f"visualizations/comparisons/{attack_params}/"
        #plot_mesh_comparison(results_df=attack_results_df, shape_path=attacked_shape_file, clean_v=original_vertices, adv_v=adv_vertices, faces=adv_faces,save_to_path=save_to_path)

        try:
            ms.add_mesh(pymeshlab.Mesh(adv_vertices, adv_faces))
            save_to_path= f"../../results/visualizations/comparisons/{attack_params}/"
            plot_mesh_comparison(results_df=attack_results_df, shape_path=attacked_shape_file, clean_v=original_vertices, adv_v=adv_vertices, faces=adv_faces,save_to_path=save_to_path)
        except Exception as e:
            print(f"Problem with {original_shape_file} (plot_mesh_comparison)")
            continue

        vertices, faces = adv_vertices, adv_faces
        normals = ms.current_mesh().face_normal_matrix()
        centers, corners, neighbors = [], [], []
        faces_contain_vertex = [set() for _ in range(len(vertices))]

        for i, (v1, v2, v3) in enumerate(faces):
            pts = vertices[[v1, v2, v3]]
            centers.append(pts.mean(axis=0))
            corners.append(pts.flatten())
            for v in (v1, v2, v3):
                faces_contain_vertex[v].add(i)

        for i, (v1, v2, v3) in enumerate(faces):
            n1 = find_neighbor(faces, faces_contain_vertex, v1, v2, i)
            n2 = find_neighbor(faces, faces_contain_vertex, v2, v3, i)
            n3 = find_neighbor(faces, faces_contain_vertex, v3, v1, i)
            neighbors.append([n1, n2, n3])
        
        shape_path = Path(attacked_shape_file)
        out_path = new_root / shape_path.relative_to(root).with_suffix('.npz')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        faces_arr = np.concatenate([np.array(centers), np.array(corners), normals], axis=1)
        np.savez(str(out_path), faces=faces_arr, neighbors=np.array(neighbors))

    print(f"Skipped {total_skiped} out of {len(attacked_shape_list)} samples.")


if __name__ == "__main__":
    global attack_results
    preprocess_df = True
    #attack_params = "_pts_2048_lowpass_spec0125"
    attack_params = "_filtered_le2048_pts_2048_lowpass_spec0125"
    dir_path = "results"
    results_file_path = f"{dir_path}/attack_results{attack_params}.csv"
    attack_results = pd.read_csv(results_file_path)
    attack_path = f"../dataset/ModelNet10_attacked{attack_params}/"
    if preprocess_df:
        attack_results['file'] = attack_results['path'].apply(lambda x: x.split('/')[-1])
        attack_results['clean_path'] = attack_results['path'].apply(lambda x: x.replace("victim_models/dataset/ModelNet10/", "../dataset/ModelNet10/"))
        attack_results['adv_path'] = attack_results['path'].apply(lambda x: x.replace("victim_models/dataset/ModelNet10/", attack_path))
        attack_results['model_fail'] = attack_results['clean'] != attack_results['true']
        attack_results['adv_success'] = attack_results['clean'] != attack_results['adv']
        allowed_suffixes = attack_results['clean_path'].apply(lambda x: x.split("../dataset/ModelNet10/")[-1].replace(".off",".npz"))
        attack_results['allowed_suffixes'] = allowed_suffixes.apply(lambda x: f"dataset/ModelNet10_processed/{x}")
        attack_results.to_csv(results_file_path,index=False)

    process_attack_to_mesh(use_full=True,attack_params=attack_params)

