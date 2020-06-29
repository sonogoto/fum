#!/usr/bin/env python3

from gcn_torch import MMGCNLayer, GCNNet, GCNConcatNet
import numpy as np
import unittest
import torch


class TestGcn(unittest.TestCase):

    def setUp(self):
        if not hasattr(self, "feature"):
            self.feature = torch.tensor(
                np.random.rand(1000, 10),
                dtype=torch.float32
            )
        if not hasattr(self, "edge_index"):
            e1 = np.random.choice(np.arange(1000), 50, replace=True).tolist()
            e2 = np.random.choice(np.arange(1000), 50, replace=True).tolist()
            self.edge_index = [e1+e2, e2+e1]

    def test_edgeidx2adjmat(self):
        MMGCNLayer.edgeidx2adjmat(self.edge_index, node_count=1000)

    def test_gcn_mp(self):
        gcn = GCNNet(
            input_dim=self.feature.size(1),
            hidden_dims=[32, 32],
            out_dim=2
        )
        _ = gcn(
            features=self.feature,
            edge=torch.tensor(self.edge_index, dtype=torch.long),
            edge_weight=torch.randn(self.edge_index[0].__len__())
        )

    def test_gcn_mm(self):
        gcn = GCNNet(
            input_dim=self.feature.size(1),
            hidden_dims=[32, 32],
            out_dim=2,
            gcn_style="mm"
        )
        adj_mat = MMGCNLayer.edgeidx2adjmat(
            edge_idx=self.edge_index,
            node_count=1000,
            edge_weight=torch.randn(self.edge_index[0].__len__()).tolist()
        )
        _ = gcn(
            features=self.feature,
            edge=adj_mat
        )

    def test_concat_gcn_mp(self):
        gcn = GCNConcatNet(
            input_dim=self.feature.size(1),
            hidden_dims=[32, 32],
            out_dim=2
        )
        _ = gcn(
            features=self.feature,
            edge=torch.tensor(self.edge_index, dtype=torch.long),
            edge_weight=torch.randn(self.edge_index[0].__len__())
        )

    def test_concat_gcn_mm(self):
        gcn = GCNConcatNet(
            input_dim=self.feature.size(1),
            hidden_dims=[32, 32],
            out_dim=2,
            gcn_style="mm"
        )
        adj_mat = MMGCNLayer.edgeidx2adjmat(
            edge_idx=self.edge_index,
            node_count=1000,
            edge_weight=torch.randn(self.edge_index[0].__len__()).tolist()
        )
        _ = gcn(
            features=self.feature,
            edge=adj_mat
        )


if __name__ == "__main__":
    unittest.main()
