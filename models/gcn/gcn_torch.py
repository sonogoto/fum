#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.sparse import mm as sparse_dense_mat_mul
from torch_geometric.nn.conv import GCNConv as MPGCNLayer
from torch_geometric.nn.inits import glorot, zeros
from utils import edgeidx2adjmat


class MMGCNLayer(nn.Module):
    """A GCN layer implemented via sparse matrix multiplication"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(MMGCNLayer, self).__init__()
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))
        self._weight_init()

    def _weight_init(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, adj_mat, edge_weight):
        """
        :param x: node attributes
        :param adj_mat: sparse adjacent matrix of graph
        :param edge_weight: only for consistency, edge weight should be encoded into `adj_mat`
        :return: the convolution result
        """
        return sparse_dense_mat_mul(
            adj_mat, torch.mm(x, self.weight)
        ) + self.bias


class GCNNet(torch.nn.Module):
    """
    A basic version GCN
    """
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 out_dim=1,
                 gcn_impl="mp",
                 **kwargs):
        """
        :param input_dim: dimension of the input features
        :param hidden_dims: list or other iterables, dimensions of the GCN layers
        :param out_dim: out dimension of the linear layer, default to 1
        :param gcn_impl: `mp` stands for massage passing style and
                            'mm' for matrix multiplication style, default to `mp`
        :param kwargs: args passed to `GCNConv.__init__`
        """
        super(GCNNet, self).__init__()
        self._input_dim = input_dim
        self._out_dim = out_dim
        if gcn_impl == "mp":
            GCNClass = MPGCNLayer
        elif gcn_impl == "mm":
            GCNClass = MMGCNLayer
        else:
            raise RuntimeError("invalid gcn_style: [%s]" % gcn_impl)
        self._gcn_layers = nn.ModuleList(
            [GCNClass(in_channels=input_dim, out_channels=hidden_dims[0], **kwargs)]
        )
        for layer_idx, size in enumerate(hidden_dims[:-1]):
            self._gcn_layers.append(
                GCNClass(in_channels=size, out_channels=hidden_dims[layer_idx+1], **kwargs)
            )
        self._fc = nn.Linear(in_features=hidden_dims[-1], out_features=out_dim)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, features, edge, edge_weight=None, mask=None):
        x = features
        for gcn_layer in self._gcn_layers:
            x = torch.tanh(gcn_layer(x, edge, edge_weight))
        if mask is not None:
            x = x[mask]
        return self._fc(x)

    def embedding(self, features, edge, layer_idx=None, mask=None):
        x = features
        for gcn_layer in self._gcn_layers[:layer_idx]:
            x = torch.tanh(gcn_layer(x, edge))
        if mask is not None:
            x = x[mask]
        return x.cpu().detach().numpy()

    @staticmethod
    def edgeidx2adjmat(edge_idx, node_count, add_self_loop=True, edge_weight=None):
        indices, values = edgeidx2adjmat(edge_idx, node_count, add_self_loop, edge_weight)
        return torch.sparse.FloatTensor(
            indices=torch.LongTensor(indices),
            values=torch.FloatTensor(values),
            size=(node_count, node_count)
        )


class GCNConcatNet(GCNNet):
    """A GCN variant with the output of adjacent layers concatenated"""
    def __init__(self,
                 input_dim,
                 hidden_dims,
                 out_dim=1,
                 gcn_impl="mp",
                 **kwargs):
        super(GCNNet, self).__init__()
        self._input_dim = input_dim
        self._out_dim = out_dim
        if gcn_impl == "mp":
            GCNClass = MPGCNLayer
        elif gcn_impl == "mm":
            GCNClass = MMGCNLayer
        else:
            raise RuntimeError("invalid gcn_style: [%s]" % gcn_impl)
        self._gcn_layers = nn.ModuleList(
            [GCNClass(in_channels=input_dim, out_channels=hidden_dims[0], **kwargs)]
        )
        for layer_idx, size in enumerate(hidden_dims[:-1]):
            self._gcn_layers.append(
                GCNClass(
                    in_channels=input_dim+sum(hidden_dims[:layer_idx+1]),
                    out_channels=hidden_dims[layer_idx + 1],
                    **kwargs
                )
            )
        self._fc = nn.Linear(in_features=input_dim+sum(hidden_dims), out_features=out_dim)

    def forward(self, features, edge, edge_weight=None, mask=None):
        x = features
        for gcn_layer in self._gcn_layers:
            x = torch.cat(
                (x, torch.relu(gcn_layer(x, edge, edge_weight))),
                dim=1
            )
        if mask is not None:
            x = x[mask]
        return self._fc(x)


class GCNIdentical(MPGCNLayer):

    def reset_parameters(self):
        self.weight.data = torch.eye(self.in_channels)
        self.bias.data.fill_(0)
        self.cached_result = None
        self.cached_num_edges = None


class StructEXGCNNet(torch.nn.Module):
    """GCN used to extract local neighbours structural of nodes"""
    def __init__(self,
                 in_dim,
                 num_layers=3,
                 **kwargs):
        """
        :param in_dim: the dimension of features
        :param num_layers: the number of gcn layers, default to 3
        :param kwargs: args passed to `GCNConv.__init__`
        """
        super(StructEXGCNNet, self).__init__()
        self._in_dim = in_dim
        self._num_layers = num_layers
        self._gcn_layers = torch.nn.ModuleList(
            [GCNIdentical(in_channels=in_dim, out_channels=in_dim, **kwargs)]
        )
        for _ in range(1, num_layers):
            self._gcn_layers.append(
                GCNIdentical(in_channels=in_dim, out_channels=in_dim, **kwargs)
            )

    def forward(self, features, edge, edge_weight=None, mask=None):
        x = features
        out = [x, ]
        for gcn_layer in self._gcn_layers:
            x = torch.relu(gcn_layer(x, edge, edge_weight))
            out.append(x)
        out = torch.cat(out, dim=1)
        if mask is not None:
            out = out[mask]
        return out


if __name__ == '__main__':
    pass

