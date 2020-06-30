#!/usr/bin/env python3


import tensorflow as tf


class FM(tf.Module):
    """Factoring Machine"""

    def __init__(
            self,
            sparse_dim,
            dense_dim=32,
            out_dim=None,
            name=None):
        """
        :param sparse_dim: sparse feature dimension
        :param dense_dim: dense vector dimension ,default to 32
        :param out_dim: out dimension, for multi-class task, set to class count, default to 1
        :param name: name of network
        """
        super(FM, self).__init__(name)
        self._sparse_dim = sparse_dim
        self._dense_dim = dense_dim
        self._out_dim = out_dim if out_dim else 1
        self._create_linear()
        self._create_interaction()

    def _create_linear(self):
        with tf.name_scope("linear"):
            self._linear_weight = tf.get_variable(
                "linear_weight",
                shape=(self._sparse_dim, self._out_dim),
                dtype=tf.float32,
                initializer=tf.glorot_normal_initializer()
            )
            self._linear_bias = tf.get_variable(
                "linear_bias",
                shape=(self._out_dim, ),
                dtype=tf.float32,
                initializer=tf.zeros_initializer()
            )

    def _create_interaction(self):
        self._dense_vectors = []
        with tf.name_scope("interaction"):
            for i in range(self._out_dim):
                self._dense_vectors.append(
                    tf.get_variable(
                        "dense_vector_%d" % i,
                        shape=(self._sparse_dim, self._dense_dim),
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.01)
                    )
                )

    @property
    def sparse_dim(self):
        return self._sparse_dim

    @property
    def dense_dim(self):
        return self._dense_dim

    @property
    def out_dim(self):
        return self._out_dim

    def __call__(self, input_x):
        linear_out = tf.add(
            tf.matmul(input_x, self._linear_weight),
            self._linear_bias
        )
        interaction_out = tf.concat(
            [self.__class__._interaction(input_x, dense_vec) for dense_vec in self._dense_vectors],
            axis=1
        )
        return tf.add(linear_out, interaction_out)

    @staticmethod
    def _interaction(input_x, dense_vec):
        square_of_sum = tf.square(tf.matmul(input_x, dense_vec))
        sum_of_square = tf.matmul(tf.square(input_x), tf.square(dense_vec))
        return tf.reduce_sum(
            tf.subtract(square_of_sum, sum_of_square),
            axis=1,
            keepdims=True
        )


if __name__ == "__main__":
    import numpy as np
    x = tf.constant(np.random.randint(10, size=(1000, 1024)) >= 9, dtype=tf.float32)
    y = FM(sparse_dim=1024, out_dim=10)(x)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(y))
