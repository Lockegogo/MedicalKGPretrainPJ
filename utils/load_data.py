import numpy as np
import scipy.sparse as sp
import torch
import pickle
import torch.nn.functional as F


def sp_coo_2_sp_tensor(sp_coo_mat):
    indices = torch.from_numpy(
        np.vstack((sp_coo_mat.row, sp_coo_mat.col)).astype(np.int64)
    )
    values = torch.from_numpy(sp_coo_mat.data)
    shape = torch.Size(sp_coo_mat.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def train_val_test_split(label_shape, train_percent):
    rand_idx = np.random.permutation(label_shape)
    val_percent = (1.0 - train_percent) / 2
    idx_train = torch.LongTensor(
        rand_idx[int(label_shape * 0.0) : int(label_shape * train_percent)]
    )
    idx_val = torch.LongTensor(
        rand_idx[
            int(label_shape * train_percent) : int(
                label_shape * (train_percent + val_percent)
            )
        ]
    )
    idx_test = torch.LongTensor(
        rand_idx[
            int(label_shape * (train_percent + val_percent)) : int(label_shape * 1.0)
        ]
    )
    return idx_train, idx_val, idx_test


def load_data(graph):
    ft_dict = dict([(k, graph.nodes[k].data['feature']) for k in graph.ntypes])

    # adj_dict 邻接矩阵
    adj_dict = dict([(k, {}) for k in graph.ntypes])
    for k in adj_dict:
        for etype in graph.canonical_etypes:
            if etype[0] == k:
                adj_dict[k][etype[2]] = graph.adj(etype=etype)

    return ft_dict, adj_dict

