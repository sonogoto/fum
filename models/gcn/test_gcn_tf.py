#!/usr/bin/env python3


from gcn_tf import GCNNet, GCNConcatNet
import unittest
import tensorflow as tf
import numpy as np


class TestGcn(unittest.TestCase):

    def setUp(self) -> None:
        if not hasattr(self, "feature"):
            self.feature = tf.constant(
                np.random.rand(1000, 10),
                dtype=tf.float32
            )
        if not hasattr(self, "edge_index"):
            e1 = np.random.choice(np.arange(1000), 50, replace=True).tolist()
            e2 = np.random.choice(np.arange(1000), 50, replace=True).tolist()
            self.edge_index = [e1+e2, e2+e1]
        self._sess = tf.Session()

    def tearDown(self) -> None:
        try:
            self._sess.close()
        except:
            pass

    def _test_core(self, cls, **kwargs):
        adj_mat = cls.edgeidx2adjmat(self.edge_index, 1000)
        out = cls(**kwargs)(self.feature, adj_mat)
        self._sess.run(tf.global_variables_initializer())
        out_np = self._sess.run(out)
        assert out_np.shape[0] == 1000
        assert out_np.shape[1] == 2

    def test_gcn(self):
        self._test_core(
            GCNNet,
            input_dim=10,
            hidden_dims=[32, 32],
            out_dim=2,
            name="gcn_net",
            activation="relu"
        )

    def test_concat_gcn(self):
        self._test_core(
            GCNConcatNet,
            input_dim=10,
            hidden_dims=[32, 32],
            out_dim=2,
            name="gcn_concat_net",
            activation="relu"
        )


if __name__ == "__mian__":
    unittest.main()
