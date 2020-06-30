#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn.parameter import Parameter


class FM(nn.Module):
    """Factoring Machine"""

    def __init__(
            self,
            sparse_dim,
            dense_dim=32,
            out_dim=None):
        """
        :param sparse_dim: sparse feature dimension
        :param dense_dim: dense vector dimension ,default to 32
        :param out_dim: out dimension, for multi-class task, set to class count, default to 1
        """
        super(FM, self).__init__()
        self._sparse_dim = sparse_dim
        self._dense_dim = dense_dim
        self._out_dim = out_dim if out_dim else 1
        self._create_linear()
        self._create_interaction()
        self._weight_init()

    def _create_linear(self):
        self._linear_weight = Parameter(
            torch.Tensor(self._sparse_dim, self._out_dim)
        )
        self._linear_bias = Parameter(
            torch.Tensor(self._out_dim)
        )

    def _create_interaction(self):
        self._dense_vectors = []
        for i in range(self._out_dim):
            self._dense_vectors.append(
                Parameter(torch.Tensor(self._sparse_dim, self._dense_dim))
            )

    def _weight_init(self):
        nn.init.xavier_normal_(self._linear_weight)
        nn.init.zeros_(self._linear_bias)
        for dense_vec in self._dense_vectors:
            nn.init.normal_(dense_vec, std=.01)

    @property
    def sparse_dim(self):
        return self._sparse_dim

    @property
    def dense_dim(self):
        return self._dense_dim

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, input_x):
        linear_out = torch.add(
            torch.matmul(input_x, self._linear_weight),
            self._linear_bias
        )
        interaction_out = torch.cat(
            [self.__class__._interaction(input_x, dense_vec) for dense_vec in self._dense_vectors],
            dim=1
        )
        return torch.add(linear_out, interaction_out)

    @staticmethod
    def _interaction(input_x, dense_vec):
        square_of_sum = torch.pow(torch.matmul(input_x, dense_vec), 2)
        sum_of_square = torch.matmul(torch.pow(input_x, 2), torch.pow(dense_vec, 2))
        return torch.sum(
            torch.sub(square_of_sum, sum_of_square),
            dim=1,
            keepdim=True
        )


if __name__ == "__main__":
    import numpy as np
    x = torch.tensor(np.random.randint(10, size=(1000, 1024)) >= 9, dtype=torch.float32)
    fm = FM(sparse_dim=1024, out_dim=8)
    y = fm(x)
    print(y)
