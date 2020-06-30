#!/usr/bin/env python3

from collections import Counter


def edgeidx2adjmat(edge_idx, node_count, add_self_loop=True, edge_weight=None):
    """directed graph not support yet"""
    node_degree = Counter(edge_idx[0])
    values = []
    for i in range(len(edge_idx[0])):
        d_i = node_degree[edge_idx[0][i]]
        d_j = node_degree[edge_idx[1][i]]
        if d_j == 0:
            raise RuntimeError("directed graph not support yet")
        weight = edge_weight[i] if edge_weight else 1
        values.append((d_i * d_j) ** -.5 * weight)
    edge_self_loop = []
    if add_self_loop:
        for i in range(node_count):
            edge_self_loop.append(i)
            if node_degree[i] == 0:
                values.append(1)
            else:
                values.append(node_degree[i] ** -.5)
    indices_0 = edge_idx[0] + edge_self_loop
    indices_1 = edge_idx[1] + edge_self_loop
    return [indices_0, indices_1], values
