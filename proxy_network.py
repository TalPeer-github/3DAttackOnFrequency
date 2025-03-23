import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RnnWalkBase(nn.Module):
    def __init__(self, params, classes, net_input_dim, model_fn=None, model_must_be_load=False,
                 dump_model_visualization=True, optimizer=None, device='cpu'):
        super(RnnWalkBase, self).__init__()

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
            checkpoint = torch.load(file_path, map_location=self.device)
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



class RnnWalkNet(RnnWalkBase):
    def __init__(self, params, classes, net_input_dim, model_fn=None, model_must_be_load=False, optimizer=None):
        if params.layer_sizes is None:
            self._layer_sizes = {
                'fc1': 64, 'fc2': 128, 'fc3': 256,
                'gru1': 1024, 'gru2': 1024, 'gru3': 512,
                'fc4': 512,
                'fc5': 128  # not used directly in current architecture
            }
        else:
            self._layer_sizes = params.layer_sizes

        super(RnnWalkNet, self).__init__(params, classes, net_input_dim, model_fn, model_must_be_load, optimizer)

    def _init_layers(self):
        self._use_norm_layer = self._params.use_norm_layer is not None

        # Normalization layers
        if self._params.use_norm_layer == 'InstanceNorm':
            self._norm1 = nn.InstanceNorm1d(self._layer_sizes['fc2'])
            self._norm2 = nn.InstanceNorm1d(self._layer_sizes['fc3'])
            self._norm3 = nn.InstanceNorm1d(self._layer_sizes['fc4'])
        elif self._params.use_norm_layer == 'BatchNorm':
            self._norm1 = nn.BatchNorm1d(self._layer_sizes['fc2'])
            self._norm2 = nn.BatchNorm1d(self._layer_sizes['fc3'])
            self._norm3 = nn.BatchNorm1d(self._layer_sizes['fc4'])

        # Fully connected layers
        self._fc1 = nn.Linear(3, self._layer_sizes['fc2'])  # Input: 3D point
        self._fc2 = nn.Linear(self._layer_sizes['fc2'], self._layer_sizes['fc3'])
        self._fc3 = nn.Linear(self._layer_sizes['fc3'], self._layer_sizes['fc4'])

        # GRU layers
        self._gru1 = nn.GRU(self._layer_sizes['fc4'] + 3, self._layer_sizes['gru1'], batch_first=True)
        self._gru2 = nn.GRU(self._layer_sizes['gru1'], self._layer_sizes['gru2'], batch_first=True)
        self._gru3 = nn.GRU(self._layer_sizes['gru2'], self._layer_sizes['gru3'], batch_first=True)

        # Final classifier
        self._fc_last = nn.Linear(self._layer_sizes['gru3'], self._classes)

    def forward(self, model_features, classify=True):
        # FC layers
        x = self._fc1(model_features)
        if self._use_norm_layer:
            x = self._norm1(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu_(x)

        x = self._fc2(x)
        if self._use_norm_layer:
            x = self._norm2(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu_(x)

        x = self._fc3(x)
        if self._use_norm_layer:
            x = self._norm3(x.transpose(1, 2)).transpose(1, 2)
        x = F.relu_(x)

        # Concatenate original coordinates with embeddings
        x = torch.cat([model_features[:, :, :3], x], dim=-1)

        # GRU layers
        x, _ = self._gru1(x)
        x, _ = self._gru2(x)
        x, _ = self._gru3(x)

        f = x[:, -1, :]  # Final time-step
        x = self._fc_last(f)

        return x  # Return raw logits (softmax applied in loss)
