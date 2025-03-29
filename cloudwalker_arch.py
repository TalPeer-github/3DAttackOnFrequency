import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CloudWalkerBase(nn.Module):
    def __init__(self, params, classes, net_input_dim, model_fn=None, model_must_be_load=False,
                 dump_model_visualization=True, optimizer=None, device='cpu'):
        super(CloudWalkerBase, self).__init__()

        self._classes = classes
        self._params = params
        self._model_must_be_load = model_must_be_load
        self.device = device

        self._init_layers()
        self.to(device)

        if dump_model_visualization:
            print(self)
            os.makedirs(self._params.logdir, exist_ok=True)
            with open(os.path.join(self._params.logdir, 'log.txt'), 'w') as f:
                f.write(str(self) + '\n')

        self.manager = None
        self.checkpoint_path = self._params.logdir

        if optimizer:
            self.optimizer = optimizer
            self.checkpoint = {"model": self.state_dict(), "optimizer": optimizer.state_dict()}

            if model_fn:
                self.load_weights(model_fn)
                self.optimizer.load_state_dict(torch.load(model_fn, map_location=device)['optimizer'])
            else:
                self.load_weights()
        else:
            self.checkpoint = {"model": self.state_dict()}
            if model_fn is None:
                model_fn = self._get_latest_model()
            self.load_weights(model_fn)

    def _print_fn(self, text):
        with open(os.path.join(self._params.logdir, 'log.txt'), 'a') as f:
            f.write(text + '\n')

    def _get_latest_model(self):
        model_files = glob.glob(os.path.join(self.checkpoint_path, '*model2keep__*.pth'))
        if not model_files:
            return None
        iters_saved = [int(f.split('model2keep__')[-1].split('.pth')[0]) for f in model_files]
        return model_files[np.argmax(iters_saved)]

    def load_weights(self, file_path=None):
        if file_path and file_path.endswith('.pth'):
            checkpoint = torch.load(file_path, map_location=self.device, weights_only=True)
            self.load_state_dict(checkpoint["model"])
            print(f"Loaded weights from {file_path}")
        elif os.path.exists(os.path.join(self.checkpoint_path, 'checkpoint.pth')):
            checkpoint = torch.load(os.path.join(self.checkpoint_path, 'checkpoint.pth'), map_location=self.device)
            self.load_state_dict(checkpoint["model"])
            print(f"Loaded checkpoint from {self.checkpoint_path}/checkpoint.pth")

    def save_weights(self, folder, step=None, keep=False, optimizer=None):
        os.makedirs(folder, exist_ok=True)
        checkpoint_path = os.path.join(folder, 'checkpoint.pth')
        torch.save({
            "model": self.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer else None
        }, checkpoint_path)

        if keep and step is not None:
            file_name = str(step).zfill(8)
            keep_path = os.path.join(folder, f'learned_model2keep__{file_name}.pth')
            torch.save({"model": self.state_dict()}, keep_path)


class CloudWalkerNet(CloudWalkerBase):
    """
    CloudWalker: A network for 3D point cloud shape analysis using random walks.
    The architecture follows the paper "CloudWalker: 3D Point Cloud Learning by Random Walks for Shape Analysis".
    
    Architecture overview:
    1. Feature extraction: 3 FC layers that process each vertex in the walk (3D → 64 → 128 → 256)
    2. Sequential processing: 3 GRU layers that aggregate walk information (1024 → 1024 → 512)
    3. Fully connected layers: 512 → 128 
    4. Classification: FC layer that maps features to class predictions
    
    References:
    - CloudWalker paper: https://arxiv.org/abs/2112.01050
    """

    def __init__(self, params, classes, net_input_dim=3, model_fn=None, model_must_be_load=False, optimizer=None):
        # Define default layer sizes if not specified in params
        if params.layer_sizes is None:
            self._layer_sizes = {
                'fc1': 64, 'fc2': 128, 'fc3': 256,  # Feature extraction layers
                'gru1': 1024, 'gru2': 1024, 'gru3': 512,  # GRU layers
                'fc4': 512, 'fc5': 128  # Post-processing FC layers
            }
        else:
            self._layer_sizes = params.layer_sizes

        super(CloudWalkerNet, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load, optimizer)

    def _init_layers(self):
        # Normalization layers
        self._use_norm_layer = self._params.use_norm_layer is not None

        if self._params.use_norm_layer == 'InstanceNorm':
            self._norm1 = nn.InstanceNorm1d(self._layer_sizes['fc2'])
            self._norm2 = nn.InstanceNorm1d(self._layer_sizes['fc3'])
        elif self._params.use_norm_layer == 'BatchNorm':
            self._norm1 = nn.BatchNorm1d(self._layer_sizes['fc2'])
            self._norm2 = nn.BatchNorm1d(self._layer_sizes['fc3'])
        
        # Initial feature extraction from 3D points
        self._vertex_proj = nn.Linear(3, self._layer_sizes['fc1'])  # 3 → 64
        self._fc1 = nn.Linear(self._layer_sizes['fc1'], self._layer_sizes['fc2'])  # 64 → 128
        self._fc2 = nn.Linear(self._layer_sizes['fc2'], self._layer_sizes['fc3'])  # 128 → 256

        # GRU layers for processing the walks sequentially
        # Concatenate the raw point coordinates with their features
        self._gru1 = nn.GRU(self._layer_sizes['fc3'] + 3, self._layer_sizes['gru1'], batch_first=True)
        self._gru2 = nn.GRU(self._layer_sizes['gru1'], self._layer_sizes['gru2'], batch_first=True)
        self._gru3 = nn.GRU(self._layer_sizes['gru2'], self._layer_sizes['gru3'], batch_first=True)
        
        # Final fully connected layers
        self._fc4 = nn.Linear(self._layer_sizes['gru3'], self._layer_sizes['fc4'])  # 512 → 512 (for consistency)
        self._fc5 = nn.Linear(self._layer_sizes['fc4'], self._layer_sizes['fc5'])  # 512 → 128

        # Classification layer
        self._classifier = nn.Linear(self._layer_sizes['fc5'], self._classes)  # 128 → num_classes
        
        # Add dropout if specified
        self._dropout = None
        if hasattr(self._params, 'dropout') and self._params.dropout > 0:
            self._dropout = nn.Dropout(self._params.dropout)

    def forward(self, walks, classify=True):
        """
        Forward pass through the CloudWalker network
        
        Args:
            walks: Tensor of shape [B, L, 3] where B is batch size and L is walk length
                   Each element is a 3D point coordinate
            classify: Whether to return class probabilities or features
            
        Returns:
            If classify=True: class probabilities
            If classify=False: (features, logits)
        """
        # Feature extraction
        x = self._vertex_proj(walks)  # [B, L, 64]
        x = self._fc1(x)  # [B, L, 128]
        if self._use_norm_layer:
            x = self._norm1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        
        x = self._fc2(x)  # [B, L, 256]
        if self._use_norm_layer:
            x = self._norm2(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu(x)
        
        # Concatenate original 3D points with their features
        x = torch.cat([walks[:, :, :3], x], dim=-1)  # [B, L, 256+3]
        
        # Process through GRU layers
        x, _ = self._gru1(x)  # [B, L, 1024]
        if self._dropout is not None:
            x = self._dropout(x)
            
        x, _ = self._gru2(x)  # [B, L, 1024]
        if self._dropout is not None:
            x = self._dropout(x)
            
        x, _ = self._gru3(x)  # [B, L, 512]
        if self._dropout is not None:
            x = self._dropout(x)
        
        # Get the last output from GRU as the walk representation
        gru_features = x[:, -1, :]  # [B, 512]
        
        # Process through final FC layers
        x = self._fc4(gru_features)  # [B, 512]
        x = F.relu(x)
        
        if self._dropout is not None:
            x = self._dropout(x)
        
        features = self._fc5(x)  # [B, 128]
        features = F.relu(features)
        
        # Classification
        logits = self._classifier(features)  # [B, num_classes]
        
        if classify:
            return F.softmax(logits, dim=1)
        else:
            return features, logits 