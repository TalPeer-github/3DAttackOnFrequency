
from easydict import EasyDict
import copy

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

import utils
from attention_model import *

import os
import random
import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import knn_graph, radius_graph


layers = tf.keras.layers


class TCL(tf.keras.Model):
    def __init__(self, n_classes=10, features_dim=10, centers=None):
        super(TCL, self).__init__()
        init = tf.initializers.Orthogonal(1)
        self.centers = tf.Variable(init(shape=(n_classes, features_dim)), name='centers')

    def call(self, features, labels, margin=1, training=None, mask=None):
        assert len(features.shape) == 2
        # TODO: add 'weighted' option for imbalanced datasets
        # features = features / tf.expand_dims(tf.norm(features, axis=-1), axis=1)
        batch_pos_centers = tf.gather(self.centers, labels)
        d_pos = tf.sqrt(tf.reduce_sum(tf.square(features - batch_pos_centers), 1))
        n_classes = self.centers.shape[0]
        neg_labels = [[x for x in range(n_classes) if not x == y] for y in range(n_classes)]
        batch_neg_centers = tf.gather(self.centers, tf.gather(neg_labels, labels))
        # batch_neg_centers = tf.stack([[self.centers[k] for k in range(self.centers.shape[0]) if not tf.math.equal(k, labels[j])] for j in range(labels.shape[0])])
        features = tf.expand_dims(features, axis=1)
        d_neg = tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.square(features - batch_neg_centers), 2)), axis=1)

        tcl = tf.reduce_mean(tf.maximum(d_pos + margin - d_neg, 0))
        return tcl

    def load_weights(self, filepath=None):
        if filepath is not None and filepath.endswith('.keras'):
            super(TCL, self).load_weights(filepath)
        else:
            filepath = filepath.replace('//', '/')
            status = self.checkpoint.restore(filepath)

    def save_weights(self, folder, step=None):
        super(TCL, self).save_weights(folder + '/TCL_centers__' + str(step).zfill(8) + '.keras')
        # self.checkpoint.write(folder + '/learned_model2keep--' + str(step))


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        dense_layer(dff, activation='elu'),  # (batch_size, seq_len, dff)
        dense_layer(d_model)  # (batch_size, seq_len, d_model)
    ])


def scaled_dot_product_attention(q, k, v, mask):
    qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention = qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention += (mask * -1e9)
    weights = tf.nn.softmax(scaled_attention, axis=-1)
    output = tf.matmul(weights, v)
    return output, weights


class CrossWalkAttention(keras.layers.Layer):
    def __init__(self, num_neurons, num_heads, out_dim, n_walks_per_model):
        super(CrossWalkAttention, self).__init__()

        self.n_walks = n_walks_per_model
        self.num_heads = num_heads
        self.num_neurons = num_neurons
        self.depth = num_neurons // self.num_heads
        self.attention_layer = scaled_dot_product_attention

        self.q_layer = dense_layer(num_neurons, initializer=None)
        self.k_layer = dense_layer(num_neurons, initializer=None)
        self.v_layer = dense_layer(num_neurons, initializer=None)

        self.linear_layer = dense_layer(out_dim, initializer=None)

    def split(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, mask):
        xs = x.shape
        if len(xs) == 2:
            x = tf.reshape(x, (-1, self.n_walks, xs[-1]))
        elif len(xs) == 3:
            x = tf.reshape(x, (-1, self.n_walks * xs[-2], xs[-1]))
            # x = tf.transpose(x, (0, 2, 1, 3))
        batch_size = tf.shape(x)[0]

        # Run through linear layers
        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)

        # Split the heads
        q = self.split(q, batch_size)
        k = self.split(k, batch_size)
        v = self.split(v, batch_size)

        # Run through attention
        # attention_output, weights = self.attention_layer(q, k, v, mask)
        new_x = tf.expand_dims(x, 1)
        attention_output, weights = self.attention_layer(new_x, new_x, v, mask)

        # Prepare for the rest of processing
        output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.num_neurons))

        output = tf.reduce_sum(tf.nn.softmax(concat_attention, axis=1) * x, axis=1)
        return output, tf.nn.softmax(concat_attention, axis=1)


class CRA2(keras.layers.Layer):
    '''
    Second try of cross-walk attention
    '''

    def _init__(self):
        super(CRA2, self)._init__()
        return

    def call(self, x):
        xs = x.shape
        if len(xs) == 2:
            x = tf.reshape(x, (-1, self.n_walks, xs[-1]))
        elif len(xs) == 3:
            x = tf.reshape(x, (-1, self.n_walks * xs[-2], xs[-1]))
            # x = tf.transpose(x, (0, 2, 1, 3))
        batch_size = tf.shape(x)[0]

        qk = tf.matmul(x, x, transpose_a=True)
        dk = tf.cast(tf.shape(x)[-1], tf.float32)
        scaled_attention = qk / tf.math.sqrt(dk)
        # if mask is not None:
        #   scaled_attention += (mask * -1e9)
        weights = tf.nn.softmax(scaled_attention, axis=-1)
        output = tf.matmul(x, weights)
        return output, weights


class SegmentationAttention(keras.layers.Layer):
    def __init__(self, num_neurons, num_heads, out_dim, n_walks_per_model):
        super(SegmentationAttention, self).__init__()

        self.n_walks = n_walks_per_model
        self.num_heads = num_heads
        self.num_neurons = num_neurons
        self.depth = num_neurons // self.num_heads
        self.attention_layer = scaled_dot_product_attention

        self.q_layer = dense_layer(num_neurons, initializer=None)
        self.k_layer = dense_layer(num_neurons, initializer=None)
        self.v_layer = dense_layer(num_neurons, initializer=None)

        self.linear_layer = dense_layer(out_dim, initializer=None)

    def split(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, mask):
        xs = x.shape
        x = tf.reshape(x, (-1, self.n_walks * xs[-2], xs[-1]))
        # x = tf.transpose(x, (0, 2, 1, 3))
        batch_size = tf.shape(x)[0]

        # Run through linear layers
        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)

        # Split the heads
        q = self.split(q, batch_size)
        k = self.split(k, batch_size)
        v = self.split(v, batch_size)

        # Run through attention
        attention_output, weights = self.attention_layer(q, k, v, mask)

        # Prepare for the rest of processing
        output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.num_neurons))

        output = tf.nn.softmax(concat_attention, axis=1) * x
        return tf.reshape(output, xs), tf.nn.softmax(concat_attention, axis=1)


class RnnWalkBase(tf.keras.Model):
    def __init__(self,
                 params,
                 classes,
                 net_input_dim,
                 model_fn=None,
                 model_must_be_load=False,
                 dump_model_visualization=True,
                 optimizer=None):
        super(RnnWalkBase, self).__init__(name='')

        self._classes = classes
        self._params = params
        self._model_must_be_load = model_must_be_load

        self._pooling_betwin_grus = 'pooling' in self._params.aditional_network_params
        self._bidirectional_rnn = 'bidirectional_rnn' in self._params.aditional_network_params

        self._init_layers()
        self.build(input_shape=(32, 800, net_input_dim))

        self.manager = None
        if optimizer:
            if model_fn:
                self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
            else:
                self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=self)
            self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self._params.logdir, max_to_keep=1)
            if model_fn:  # Transfer learning
                self.load_weights(model_fn)
                self.checkpoint.optimizer = optimizer
            else:
                self.load_weights()
        else:
            self.checkpoint = tf.train.Checkpoint(model=self)
            if model_fn:
                self.load_weights(model_fn)
            else:
                self.load_weights(tf.train.latest_checkpoint(self._params.logdir))

    def _print_fn(self, st):
        with open(self._params.logdir + '/log.txt', 'at') as f:
            f.write(st + '\n')

    def load_weights(self, filepath=None):
        if filepath is not None and filepath.endswith('.keras'):
            if hasattr(self._params, 'train_multiwalk_head') and self._params.train_multiwalk_head:
                super(RnnWalkBase, self).load_weights(filepath, by_name=True, skip_mismatch=True)
            else:
                super(RnnWalkBase, self).load_weights(filepath)
        elif filepath is None:
            status = self.checkpoint.restore(self.manager.latest_checkpoint)
            print(utils.color.BLUE, 'Starting from iteration: ', self.checkpoint.optimizer.iterations.numpy(),
                  utils.color.END)
        else:
            filepath = filepath.replace('//', '/')
            status = self.checkpoint.restore(filepath)

    def save_weights(self, folder, step=None, keep=False):
        if self.manager is not None:
            self.manager.save()
        if keep:
            super(RnnWalkBase, self).save_weights(folder + '/learned_model2keep__' + str(step).zfill(8) + '.keras')
            # self.checkpoint.write(folder + '/learned_model2keep--' + str(step))


class RnnWalkNet(RnnWalkBase):
    def __init__(self,
                 params,
                 classes,
                 net_input_dim,
                 model_fn,
                 model_must_be_load=False,
                 dump_model_visualization=True,
                 optimizer=None):
        # TODO: add cross_walk_attn / in_walk_attn as variables
        if params.layer_sizes is None:
            self._layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 512, 'gru2': 512, 'gru3': 256}
        else:
            self._layer_sizes = params.layer_sizes
        self.cross_walk_attn = params.cross_walk_attn if hasattr(params, 'cross_walk_attn') else False
        self.in_walk_attn = params.in_walk_attn if hasattr(params, 'in_walk_attn') else False
        super(RnnWalkNet, self).__init__(params, classes, net_input_dim, model_fn,
                                         model_must_be_load=model_must_be_load,
                                         dump_model_visualization=dump_model_visualization, optimizer=optimizer)

    def _init_layers(self):
        kernel_regularizer = tf.keras.regularizers.l2(0.0001)
        initializer = tf.initializers.Orthogonal(3)

        self._use_norm_layer = self._params.use_norm_layer is not None
        if self._params.use_norm_layer == 'InstanceNorm':
            self._norm1 = tfa.layers.InstanceNormalization(axis=2)
            self._norm2 = tfa.layers.InstanceNormalization(axis=2)
        elif self._params.use_norm_layer == 'BatchNorm':
            self._norm1 = layers.BatchNormalization(axis=2)
            self._norm2 = layers.BatchNormalization(axis=2)
        self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
        self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)

        # self.embedder = MHA_embedder(self._layer_sizes['fc1'], self._layer_sizes['fc2'], 4, self._params.seq_len // 10)
        # rnn_layer = layers.LSTM
        rnn_layer = layers.GRU
        self._gru1 = rnn_layer(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                               # trainable=False,
                               # activation='sigmoid',
                               dropout=self._params.net_gru_dropout,
                               # recurrent_dropout=self._params.net_gru_dropout,      --->> very slow!! (tf2.1)
                               recurrent_initializer=initializer, kernel_initializer=initializer,
                               kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                               bias_regularizer=kernel_regularizer)
        if self._bidirectional_rnn:
            self._gru1 = layers.Bidirectional(self._gru1)
        self._gru2 = rnn_layer(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                               # trainable=False,
                               # activation='sigmoid',
                               dropout=self._params.net_gru_dropout,
                               # recurrent_dropout=self._params.net_gru_dropout,
                               recurrent_initializer=initializer, kernel_initializer=initializer,
                               kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                               bias_regularizer=kernel_regularizer)
        if self._bidirectional_rnn:
            self._gru2 = layers.Bidirectional(self._gru2)
        self._gru3 = rnn_layer(self._layer_sizes['gru3'], time_major=False,
                               return_sequences=not self._params.one_label_per_model,
                               return_state=False,
                               # trainable=False,
                               # activation='sigmoid',
                               dropout=self._params.net_gru_dropout,
                               # recurrent_dropout=self._params.net_gru_dropout,
                               recurrent_initializer=initializer, kernel_initializer=initializer,
                               kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                               bias_regularizer=kernel_regularizer)
        if self._bidirectional_rnn:
            self._gru3 = layers.Bidirectional(self._gru3)
            print('Using Bidirectional GRUs.')
        self._fc_last = layers.Dense(self._classes, activation=self._params.last_layer_actication,
                                     kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                                     kernel_initializer=initializer)
        self._pooling = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')

        self._norm_input = False
        if self._norm_input:
            self._norm_features = layers.LayerNormalization(axis=-1, trainable=False)

        if self.cross_walk_attn:
            layer = CrossWalkAttention if self._params.one_label_per_model else SegmentationAttention
            self._cw_attn_layer = layer(self._layer_sizes['gru3'], 1, 1, self._params.n_walks_per_model)

    # @tf.function
    def call(self, model_ftrs, classify=True, skip_1st=False, training=True, mask=None):
        if self._norm_input:
            model_ftrs = self._norm_features(model_ftrs)
        if skip_1st:
            x = model_ftrs[:, 1:]
        else:
            x = model_ftrs
        x = self._fc1(x)
        x = self._norm1(x, training=training)
        x = tf.nn.relu(x)
        x = self._fc2(x)
        x = self._norm2(x, training=training)
        x = tf.nn.relu(x)
        # x = self.embedder(x, training=training)

        x1 = self._gru1(x, training=training)
        if self._pooling_betwin_grus:
            x1 = self._pooling(x1)
            if mask is not None:
                mask = mask[:, ::2]
        x2 = self._gru2(x1, training=training)
        if self._pooling_betwin_grus:
            x2 = self._pooling(x2)
            if mask is not None:
                mask = mask[:, ::2]
        x3 = self._gru3(x2, training=training, mask=mask)
        features = x3

        if self.cross_walk_attn:
            features, weights = self._cw_attn_layer(features, None)

        # if self._params.one_label_per_model:
        #  x = x[:, -1, :]
        x = self._fc_last(features)
        if classify == 'both':
            return x, features
        if classify == 'weights':
            return x, weights, features
        if classify == 'visualize':
            return x, weights, self._fc_last(x3)
        return x

    def call_dbg(self, model_ftrs, classify=True, skip_1st=True, training=True, get_layer=None):
        if skip_1st:
            x = model_ftrs[:, 1:]
        else:
            x = model_ftrs
        if get_layer == 'input':
            return x
        x = self._fc1(x)
        if self._use_norm_layer:
            x = self._norm1(x, training=training)
        x = tf.nn.relu(x)
        if get_layer == 'fc1':
            return x
        x = self._fc2(x)
        if self._use_norm_layer:
            x = self._norm2(x, training=training)
        x = tf.nn.relu(x)
        if get_layer == 'fc2':
            return x
        x = self._gru1(x, training=training)
        if get_layer == 'gru1':
            return x
        x = self._gru2(x, training=training)
        if get_layer == 'gru2':
            return x
        x = self._gru3(x, training=training)
        if get_layer == 'gru3':
            return x

        if self._params.one_label_per_model:
            x = x[:, -1, :]

        if classify:
            x = self._fc_last(x)
        return x


class MultiwalkHead(RnnWalkBase):
    def __init__(self,
                 params,
                 classes,
                 net_input_dim,
                 model_fn,
                 model_must_be_load=False,
                 dump_model_visualization=True,
                 optimizer=None):
        # TODO: add cross_walk_attn / in_walk_attn as variables
        if params.layer_sizes is None:
            self._layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512}
        else:
            self._layer_sizes = params.layer_sizes
        super(MultiwalkHead, self).__init__(params, classes, net_input_dim, model_fn,
                                            model_must_be_load=model_must_be_load,
                                            dump_model_visualization=dump_model_visualization, optimizer=optimizer)


# ===== Attention for attentive-GRU ===== #
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


# ============= End of bahadanu attention tutorial code tensorflow -


class AttentionWalkNet(RnnWalkBase):
    def __init__(self,
                 params,
                 classes,
                 net_input_dim,
                 model_fn,
                 layer_sizes={'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512, 'gru_dec1': 512,
                              'gru_dec2': 512},
                 model_must_be_load=False,
                 optimizer=None):
        self._layer_sizes = layer_sizes
        super(AttentionWalkNet, self).__init__(params, classes, net_input_dim, model_fn,
                                               model_must_be_load=model_must_be_load, optimizer=optimizer)

    def _init_layers(self):
        kernel_regularizer = tf.keras.regularizers.l2(0.0001)
        initializer = tf.initializers.Orthogonal(3)
        self._use_norm_layer = 1
        if self._use_norm_layer:
            self._norm1 = tfa.layers.InstanceNormalization(axis=2)
            self._norm2 = tfa.layers.InstanceNormalization(axis=2)
        self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
        self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
        # self.embedder = MHA_embedder(self._layer_sizes['fc1'], self._layer_sizes['fc2'], 4, self._params.seq_len)
        self._gru1 = layers.GRU(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                                # activation='sigmoid',
                                recurrent_initializer=initializer, kernel_initializer=initializer,
                                kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                                bias_regularizer=kernel_regularizer)
        self._gru2 = layers.GRU(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                                # activation='sigmoid',
                                recurrent_initializer=initializer, kernel_initializer=initializer,
                                kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                                bias_regularizer=kernel_regularizer)

        self._gru3 = layers.GRU(self._layer_sizes['gru3'], time_major=False, return_sequences=True, return_state=True,
                                # activation='sigmoid',
                                recurrent_initializer=initializer, kernel_initializer=initializer,
                                kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                                bias_regularizer=kernel_regularizer)
        self._attention_layer = BahdanauAttention(self._layer_sizes['gru_dec1'])

        self._gru_decode_1 = layers.GRU(self._layer_sizes['gru_dec1'], time_major=False, return_sequences=True,
                                        return_state=False,
                                        recurrent_initializer=initializer, kernel_initializer=initializer,
                                        kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                                        bias_regularizer=kernel_regularizer)
        self._gru_decode_2 = layers.GRU(self._layer_sizes['gru_dec2'], time_major=False, return_sequences=True,
                                        return_state=False,
                                        recurrent_initializer=initializer, kernel_initializer=initializer,
                                        kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                                        bias_regularizer=kernel_regularizer)

        self._fc_last = layers.Dense(self._classes, activation='softmax', kernel_regularizer=kernel_regularizer,
                                     bias_regularizer=kernel_regularizer,
                                     kernel_initializer=initializer)

    @tf.function
    def call(self, model_ftrs, classify=True, skip_1st=True, training=True):
        if skip_1st:
            x = model_ftrs[:, 1:]
        else:
            x = model_ftrs
        # model_ftrs_ = x

        # Encoder
        # -------
        x = self._fc1(x)
        if self._use_norm_layer:
            x = self._norm1(x, training=training)
        x = tf.nn.relu(x)
        x = self._fc2(x)
        if self._use_norm_layer:
            x = self._norm2(x, training=training)
        x_embedded = tf.nn.relu(x)

        x = self._gru1(x_embedded)
        x = self._gru2(x)
        output, hidden = self._gru3(x)

        # Attention
        # ---------
        context_vector, attention_weights = self._attention_layer(hidden, output)

        # Decoder
        # -------
        x = tf.concat([tf.expand_dims(context_vector, 1), x_embedded], axis=-1)
        x = self._gru_decode_1(x)
        x = self._gru_decode_2(x)
        x = self._fc_last(x)

        return x


class RnnStrideWalkNet(RnnWalkBase):
    def __init__(self,
                 params,
                 classes,
                 net_input_dim,
                 model_fn,
                 layer_sizes={'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 1024},
                 model_must_be_load=False):
        self._layer_sizes = layer_sizes
        super(RnnStrideWalkNet, self).__init__(params, classes, net_input_dim, model_fn,
                                               model_must_be_load=model_must_be_load)

    def _init_layers(self):
        kernel_regularizer = tf.keras.regularizers.l2(0.0001)
        initializer = tf.initializers.Orthogonal(3)
        self._use_norm_layer = 1
        if self._use_norm_layer:
            self._norm1 = tfa.layers.InstanceNormalization(axis=2)
            self._norm2 = tfa.layers.InstanceNormalization(axis=2)
        self._fc1 = layers.Dense(self._layer_sizes['fc1'], kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
        self._fc2 = layers.Dense(self._layer_sizes['fc2'], kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=kernel_regularizer,
                                 kernel_initializer=initializer)
        self._gru1 = layers.GRU(self._layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                                # activation='sigmoid',
                                recurrent_initializer=initializer, kernel_initializer=initializer,
                                kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                                bias_regularizer=kernel_regularizer)
        self._gru2 = layers.GRU(self._layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                                # activation='sigmoid',
                                recurrent_initializer=initializer, kernel_initializer=initializer,
                                kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                                bias_regularizer=kernel_regularizer)
        self._gru3 = layers.GRU(self._layer_sizes['gru3'], time_major=False, return_sequences=True, return_state=False,
                                # activation='sigmoid',
                                recurrent_initializer=initializer, kernel_initializer=initializer,
                                kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                                bias_regularizer=kernel_regularizer)
        self._fc_last = layers.Dense(self._classes, activation='sigmoid', kernel_regularizer=kernel_regularizer,
                                     bias_regularizer=kernel_regularizer,
                                     kernel_initializer=initializer)
        self._pooling = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')
        self._up_sampling = layers.UpSampling1D(size=2)

    # @tf.function
    def call(self, model_ftrs, classify=True, skip_1st=True, training=True):
        if skip_1st:
            x = model_ftrs[:, 1:]
        else:
            x = model_ftrs
        x = self._fc1(x)
        if self._use_norm_layer:
            x = self._norm1(x, training=training)
        x = tf.nn.relu(x)
        x = self._fc2(x)
        if self._use_norm_layer:
            x = self._norm2(x, training=training)
        x = tf.nn.relu(x)
        x = self._gru1(x)
        before_pooling = x
        x = self._pooling(x)
        x = self._gru2(x)
        x = self._gru3(x)
        x = self._up_sampling(x)
        x = x[:, :before_pooling.shape[1], :] + before_pooling

        if self._params.one_label_per_model:
            x = x[:, -1, :]

        if classify:
            x = self._fc_last(x)
        return x


def set_up_rnn_walk_model():
    _layer_sizes = {'fc1': 128, 'fc2': 256, 'gru1': 1024, 'gru2': 1024, 'gru3': 512}
    last_layer_actication = 'softmax'
    _classes = 40
    training = True
    one_label_per_model = True
    classify = True

    input = keras.Input(shape=(28, 28, 1), name='original_img')

    kernel_regularizer = tf.keras.regularizers.l2(0.0001)
    initializer = tf.initializers.Orthogonal(3)
    _norm1 = tfa.layers.InstanceNormalization(axis=2)
    _norm2 = tfa.layers.InstanceNormalization(axis=2)
    _fc1 = layers.Dense(_layer_sizes['fc1'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                        kernel_initializer=initializer)
    _fc2 = layers.Dense(_layer_sizes['fc2'], kernel_regularizer=kernel_regularizer, bias_regularizer=kernel_regularizer,
                        kernel_initializer=initializer)
    _gru1 = layers.GRU(_layer_sizes['gru1'], time_major=False, return_sequences=True, return_state=False,
                       recurrent_initializer=initializer, kernel_initializer=initializer,
                       kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                       bias_regularizer=kernel_regularizer)
    _gru2 = layers.GRU(_layer_sizes['gru2'], time_major=False, return_sequences=True, return_state=False,
                       recurrent_initializer=initializer, kernel_initializer=initializer,
                       kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                       bias_regularizer=kernel_regularizer)
    _gru3 = layers.GRU(_layer_sizes['gru3'], time_major=False, return_sequences=True, return_state=False,
                       recurrent_initializer=initializer, kernel_initializer=initializer,
                       kernel_regularizer=kernel_regularizer, recurrent_regularizer=kernel_regularizer,
                       bias_regularizer=kernel_regularizer)
    _fc_last = layers.Dense(_classes, activation=last_layer_actication, kernel_regularizer=kernel_regularizer,
                            bias_regularizer=kernel_regularizer,
                            kernel_initializer=initializer)

    inputs = keras.Input(shape=(100, 4,))
    x = inputs
    x = _fc1(x)
    x = _norm1(x, training=training)
    x = tf.nn.relu(x)
    x = _fc2(x)
    x = _norm2(x, training=training)
    x = tf.nn.relu(x)
    x = _gru1(x)
    x = _gru2(x)
    x = _gru3(x)

    if one_label_per_model:
        x = x[:, -1, :]

    if classify:
        x = _fc_last(x)

    outputs = x

    model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')

    return model


def show_model():
    def fn(to_print):
        print(to_print)

    if 1:
        params = EasyDict({'n_classes': 3, 'net_input_dim': 3, 'batch_size': 32, 'last_layer_actication': 'softmax',
                           'one_label_per_model': True, 'logdir': '.'})
        params.net_input_dim = 3 + 5
        model = RnnWalkNet(params, classes=3, net_input_dim=3, model_fn=None)
    else:
        model = set_up_rnn_walk_model()
        tf.keras.utils.plot_model(model, "RnnWalkModel.png", show_shapes=True)
        model.summary(print_fn=fn)


if __name__ == '__main__':
    np.random.seed(0)
    utils.config_gpu(0)
    show_model()
