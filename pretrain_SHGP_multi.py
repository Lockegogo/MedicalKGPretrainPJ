import argparse

from loader import BioDataset
from util import ScorePredictor
from model import GNN

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import sklearn.linear_model as lm
import sklearn.metrics as skm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
import numpy as np
import pandas as pd
import os

from utils import load_data, set_params
from utils.evaluate import evaluate
from utils.cluster import kmeans
from module.att_lpa import *
from module.att_hgcn import ATT_HGCN
import warnings

warnings.filterwarnings('ignore')
import pickle as pkl

import random
import time

import matplotlib.pyplot as plt


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description='DGL implementation of pre-training of medical KG'
    )
    parser.add_argument(
        '--device', type=int, default=0, help='which gpu to use if any (default: 0)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='input batch size for training (default: 1024)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs to train (default: 6)',
    )
    parser.add_argument('--warm_epochs', type=int, default=10)
    parser.add_argument(
        '--lr', type=float, default=0.001, help='learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--decay', type=float, default=0, help='weight decay (default: 0)'
    )
    parser.add_argument(
        '--num_layer',
        type=int,
        default=2,
        help='number of GNN message passing layers (default: 2).',
    )
    parser.add_argument(
        '--emb_dim', type=int, default=256, help='embedding dimensions (default: 256)'
    )
    parser.add_argument(
        '--dropout_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)'
    )
    parser.add_argument(
        '--neg_samples',
        type=int,
        default=1,
        help='number of negative contexts per positive context (default: 1)',
    )
    parser.add_argument(
        '--context_pooling',
        type=str,
        default="mean",
        help='how the contexts are pooled (sum, mean, or max)',
    )
    parser.add_argument('--gnn_type', type=str, default="GAT")

    parser.add_argument('--mode', type=str, default="cbow", help="cbow or skipgram")

    parser.add_argument('--use_info', type=str, default=False)
    parser.add_argument('--use_linkpred_emb', type=str, default=True)

    parser.add_argument('--compress_ratio', type=int, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=[256, 256])
    parser.add_argument('--type_fusion', type=str, default='att')
    parser.add_argument('--type_att_size', type=int, default=64)

    parser.add_argument(
        '--seed', type=int, default=42, help="Seed for splitting dataset."
    )
    parser.add_argument(
        '--runseed', type=int, default=0, help="Seed for running experiments."
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='number of workers for dataset loading',
    )
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    dgl.random.seed(args.runseed)

    # set up dataset
    dataset_path = 'data/BioKG'
    dataset = BioDataset(dPath=dataset_path)
    graph, idx_node_map, idx_node_id_map = dataset.to_graph(
        emb_dim=args.emb_dim,
        use_info=args.use_info,
        use_linkpred_emb=args.use_linkpred_emb,
    )

    ft_dict, adj_dict = load_data(graph)

    target_type = 'PROTEIN'
    num_cluster = int(
        ft_dict[target_type].shape[0] * args.compress_ratio
    )  # compress the range of

    # initial pseudo-labels.
    init_pseudo_label = 0
    pseudo_pseudo_label = 0

    layer_shape = []
    input_layer_shape = dict([(k, ft_dict[k].shape[1]) for k in ft_dict.keys()])
    hidden_layer_shape = [
        dict.fromkeys(ft_dict.keys(), l_hid) for l_hid in args.hidden_dim
    ]
    output_layer_shape = dict.fromkeys(ft_dict.keys(), num_cluster)

    layer_shape.append(input_layer_shape)
    layer_shape.extend(hidden_layer_shape)
    layer_shape.append(output_layer_shape)

    net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
    node_types = graph.ntypes
    model = ATT_HGCN(
        net_schema=net_schema,
        layer_shape=layer_shape,
        # label_keys=['PROTEIN'],
        label_keys=node_types,
        type_fusion=args.type_fusion,
        type_att_size=args.type_att_size,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    model.to(device)
    for k in ft_dict:
        ft_dict[k] = ft_dict[k].to(device)
    for k in adj_dict:
        for kk in adj_dict[k]:
            adj_dict[k][kk] = adj_dict[k][kk].to(device)

    best = 1e9
    loss_list = []

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits, embd, attention_dict = model(ft_dict, adj_dict)
        target_embd = embd[target_type]

        if epoch == 0:
            init_pseudo_label = init_lpa(
                adj_dict, ft_dict, target_type, num_cluster, device
            )
            pseudo_label_dict = init_pseudo_label
        elif epoch < args.warm_epochs:
            pseudo_label_dict = init_pseudo_label
        else:
            pseudo_label_dict = att_lpa(
                adj_dict,
                init_pseudo_label,
                attention_dict,
                target_type,
                num_cluster,
                device,
            )
            init_pseudo_label = pseudo_label_dict

        # label_predict = torch.argmax(pseudo_label_dict[target_type], dim=1)
        # logits = F.log_softmax(logits[target_type], dim=1)
        # loss_train = F.nll_loss(logits, label_predict.long().detach())

        ## consider all types of nodes
        label_predict = dict(
            [(k, torch.argmax(pseudo_label_dict[k], dim=1)) for k in node_types]
        )
        logits = dict([(k, F.log_softmax(logits[k], dim=1)) for k in node_types])
        loss_train = 0
        for k in node_types:
            loss_train += F.nll_loss(logits[k], label_predict[k].long().detach())

        loss_train.backward()
        optimizer.step()
        loss_list.append(loss_train.item())
        if loss_train < best:
            best = loss_train

        print(
            'epoch: {:3d}'.format(epoch),
            'train loss: {:.4f}'.format(loss_train.item()),
        )

    # evaluate
    _, embd, _ = model(ft_dict, adj_dict)

    # # get "PROTEIN" embedding
    # target_embd = embd["PROTEIN"]

    # save emb
    save_emb(embd, idx_node_map, idx_node_id_map, args.epochs, args.use_linkpred_emb)

    # train visualization
    plot(loss_list, args.epochs, args.use_linkpred_emb)


def save_emb(emb, idx_node_map, idx_node_id_map, epoch, use_linkpred_emb):
    node_feature_dict = {}

    for ntype in emb.keys():
        node_feature_dict[ntype] = {}
        for i in range(emb[ntype].shape[0]):
            node_name = idx_node_id_map[ntype][i]
            node_feature_dict[ntype][node_name] = emb[ntype][i].cpu().tolist()

    if use_linkpred_emb:
        emb_name = "pretrained_emb_dict_linkpred_SHGP_" + str(epoch) + '.pkl'
    else:
        emb_name = "pretrained_emb_dict_SHGP_" + str(epoch) + '.pkl'

    emb_path = os.path.join('pretrain_emb', emb_name)
    with open(emb_path, 'wb') as f:
        pkl.dump(node_feature_dict, f, pkl.HIGHEST_PROTOCOL)


def plot(train_loss, epoch, use_linkpred_emb):
    # 创建画布
    plt.figure()
    plt.plot(
        np.arange(len(train_loss)), np.array(train_loss), c="blue", label="train loss"
    )
    plt.xlabel("train epochs", fontsize=13)
    plt.ylabel("train loss", fontsize=13)
    plt.legend()

    if use_linkpred_emb:
        fig_name = "results/linkpred_SHGP_train_" + str(epoch) + '.png'
    else:
        fig_name = "results/SHGP_train_" + str(epoch) + '.png'

    plt.savefig(fig_name)


if __name__ == "__main__":
    main()

