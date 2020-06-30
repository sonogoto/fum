#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import activations
from utils import edgeidx2adjmat


class GCNLayer(tf.Module):

    def __init__(self, in_dim, out_dim, name=None, activation=None):
        super(GCNLayer, self).__init__(name)
        self.weight = tf.Variable(
            tf.initializers.glorot_normal()(shape=(in_dim, out_dim))
        )
        self.bias = tf.Variable(
            tf.initializers.zeros()(shape=(out_dim,))
        )
        self.activation = activations.get(activation)

    def __call__(self, x, adj_mat):
        y = tf.sparse_tensor_dense_matmul(
            adj_mat,
            tf.matmul(x, self.weight)
        )
        return self.activation(y)


class GCNNet(tf.Module):
    """A basic version GCN"""

    def __init__(self,
                 input_dim,
                 hidden_dims,
                 out_dim=1,
                 name=None,
                 activation=None):
        """
        :param input_dim: dimension of the input features
        :param hidden_dims: list or other iterables, dimensions of the GCN layers
        :param out_dim: out dimension of the linear layer, default to 1
        :param name: name of GCN net
        :param activation: active function of GCN layer
        """
        super(GCNNet, self).__init__(name)
        self._input_dim = input_dim
        self._out_dim = out_dim
        self._gcn_layers = [
            GCNLayer(in_dim=input_dim, out_dim=hidden_dims[0], name="gcn_1", activation=activation)
        ]
        for layer_idx, size in enumerate(hidden_dims[:-1]):
            self._gcn_layers.append(
                GCNLayer(
                    in_dim=size,
                    out_dim=hidden_dims[layer_idx + 1],
                    name="gcn_%d" % (layer_idx+2),
                    activation=activation
                )
            )
        self._fc = tf.keras.layers.Dense(units=out_dim, name="fc")

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def out_dim(self):
        return self._out_dim

    def __call__(self, features, adj_mat, mask=None):
        x = features
        for layer in self._gcn_layers:
            x = layer(x, adj_mat)
        if mask is not None:
            x = x[mask]
        return self._fc(x)

    def embedding(self, features, adj_mat, mask=None):
        x = features
        for layer in self._gcn_layers:
            x = layer(x, adj_mat)
        if mask is not None:
            x = x[mask]
        return x

    @staticmethod
    def edgeidx2adjmat(edge_idx, node_count, add_self_loop=True, edge_weight=None):
        indices, values = edgeidx2adjmat(edge_idx, node_count, add_self_loop, edge_weight)
        return tf.SparseTensor(
            indices=[[i, j] for i, j in zip(*indices)],
            values=values,
            dense_shape=(node_count, node_count)
        )


class GCNConcatNet(GCNNet):
    """A GCN variant with the output of adjacent layers concatenated"""
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 out_dim=1,
                 name=None,
                 activation=None):
        super(GCNNet, self).__init__(name)
        self._input_dim = input_dim
        self._out_dim = out_dim
        self._gcn_layers = [
            GCNLayer(in_dim=input_dim, out_dim=hidden_dims[0], name="gcn_1", activation=activation)
        ]
        for layer_idx, size in enumerate(hidden_dims[:-1]):
            self._gcn_layers.append(
                GCNLayer(
                    in_dim=input_dim+sum(hidden_dims[:layer_idx+1]),
                    out_dim=hidden_dims[layer_idx + 1],
                    name="gcn_%d" % (layer_idx+2),
                    activation=activation
                )
            )
        self._fc = tf.keras.layers.Dense(units=out_dim, name="fc")

    def __call__(self, features, adj_mat, mask=None):
        x = features
        for layer in self._gcn_layers:
            x = tf.concat(
                (x, layer(x, adj_mat)),
                axis=1
            )
        if mask is not None:
            x = x[mask]
        return self._fc(x)

    def embedding(self, features, adj_mat, mask=None):
        x = features
        for layer in self._gcn_layers:
            x = tf.concat(
                (x, layer(x, adj_mat)),
                axis=1
            )
        if mask is not None:
            x = x[mask]
        return x


if __name__ == "__main__":
    pass
