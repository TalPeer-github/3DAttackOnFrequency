import os
import os.path as osp
import yaml

    
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



def _check_dir(dir, make_dir=True):
    if not osp.exists(dir):
        if make_dir:
            print('Create directory {}'.format(dir))
            os.mkdir(dir)
        else:
            raise Exception('Directory not exist: {}'.format(dir))


def get_train_config(model_type='mesh',config_file='config/mesh_train_config.yaml'):
    if model_type == 'mesh':
        config_file = 'config/mesh_train_config.yaml'
        with open(config_file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)

        _check_dir(cfg['dataset']['data_root'], make_dir=False)
        _check_dir(cfg['ckpt_root'])

        return cfg
    
    elif model_type == 'pc':
        config_file = 'config/pc_train_config.yaml'
        with open(config_file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)

        _check_dir(cfg['dataset']['data_root'], make_dir=False)
        _check_dir(cfg['ckpt_root'])

        return cfg


def get_test_config(model_type, config_file='config/test_config.yaml'):
    if model_type == 'attack':
        config_file = 'config/attack_config.yaml'
        
        with open(config_file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)

        _check_dir(cfg['dataset']['data_root'], make_dir=False)

        return cfg  

    elif model_type == 'pc':
        config_file = 'config/pc_test_config.yaml'
        
        with open(config_file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)

        _check_dir(cfg['dataset']['data_root'], make_dir=False)

        return cfg
    
    else: # mesh
        config_file = 'config/mesh_test_config.yaml'
        
        with open(config_file, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)

        _check_dir(cfg['dataset']['data_root'], make_dir=False)

        return cfg