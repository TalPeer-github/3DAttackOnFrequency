from easydict import EasyDict
import numpy as np

import utils


def get_seq_random_walk_local_jumps(mesh_extra, f0, seq_len):
    """
    for an input set S_i we generate the walk W_ij of length l as follows:
    1. the walk origin point, p0, is randmoly selected from the set S_i.
    2. point are iteratively added to the walk by selecting random point from the set of k-nearest neighbors of the last
        point in the sequence (exluding neighbors who were added at an earlier stage)
    3.  In the rare case where all k nearest neighbors were already added to the walk, a new
        random un-visited point is chosen and the walk generation proceeds as before.
    Choosing the closest neighbor imposes a strong constraint on the generation process
    and reduces the randomness and the ability to visit sparser regions.

     KDtree is an efficient, hierarchical space partitioning data structure for nearest neighbors queries,
     in which every node represents an axis-aligned hyper-rectangle and contains the set of points in it.
    :param mesh_extra: 
    :param f0: 
    :param seq_len: 
    :return: 
    """
    n_vertices = mesh_extra['kdtree_query'].shape[0]
    kdtr = mesh_extra['kdtree_query']

    seq = np.zeros((seq_len + 1,), dtype=np.int32) - 1
    jumps = np.zeros((seq_len + 1,), dtype=np.bool)
    seq[0] = f0
    visited = np.zeros((n_vertices + 1,), dtype=np.bool)
    visited[-1] = True
    visited[f0] = True
    for i in range(1, seq_len + 1):
        to_consider = [n for n in kdtr[seq[i - 1]] if not visited[n]]
        if len(to_consider):
            random_point = np.random.choice(to_consider)
            seq[i] = random_point
            jumps[i] = False
        else:
            seq[i] = np.random.randint(n_vertices)
            jumps[i] = True
            visited = np.zeros((n_vertices + 1,), dtype=np.bool)
            visited[-1] = True
        visited[seq[i]] = True

    return seq, jumps


def get_model():
    from dataset_prepare import prepare_kdtree
    from dataset import load_model_from_npz

    model_fn = 'datasets_processed/modelnet40_normal_resampled/test__5000__toilet__toilet_0399.npz'
    model = load_model_from_npz(model_fn)
    model_dict = EasyDict({'vertices': np.asarray(model['vertices']), 'n_vertices': model['vertices'].shape[0],
                           'vertex_normals': np.asarray(model['vertex_normals'])})
    model_dict['vertices'] = model_dict['vertices'][0:1024]
    model_dict['vertex_normals'] = model_dict['vertex_normals'][0:1024]
    model_dict['n_vertices'] = 1024
    prepare_kdtree(model_dict)
    return model_dict


def show_walk_on_model(seed=0):
    walks = []
    vertices = model['vertices']
    coverage_value = np.array([0] * model['n_vertices'])
    for i in range(1):
        f0 = np.random.randint(model['n_vertices'])
        walk, jumps = get_seq_random_walk_local_jumps(model, f0, 500)
        coverage_value[walk] = 1
        walks.append(walk)
    utils.visualize_model_walk(vertices, walks, seed=seed)


if __name__ == '__main__':
    utils.config_gpu(False)
    model = get_model()
    seed = 1969
    np.random.seed(seed)
    show_walk_on_model(seed)
