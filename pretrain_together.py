import argparse
import os
import warnings

import dgl
import numpy as np
import torch

from layers import *
from loader import BioDataset

warnings.filterwarnings('ignore')
import pickle as pkl
from collections import defaultdict

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
        default=1000,
        help='number of epochs to train (default: 6)',
    )
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

    # cooperation
    parser.add_argument('--gnn_type', type=str, default="GAT")
    parser.add_argument('--use_info', type=str, default=False)
    # 初始化很重要，如果不初始化的话，最后训练出来的 emb 大多都是 0，不知道怎么回事
    parser.add_argument('--use_linkpred_emb', type=str, default=True)
    parser.add_argument('--use_SRRSC_emb', type=str, default=False)

    # SRRSC
    parser.add_argument('--hid_units', type=int, default=256)
    parser.add_argument('--hid_units2', type=int, default=256)
    parser.add_argument('--out_ft', type=int, default=256)
    parser.add_argument('--isAtt', action='store_true', default=True)
    parser.add_argument(
        # margin 越大，不同类别之间的区分越强！
        '--margin',
        type=float,
        default=0.8,
        help='coefficient for the margin loss',
    )
    parser.add_argument('--drop_prob', type=float, default=0.0)
    parser.add_argument(
        '--lamb', type=float, default=1, help='coefficient for the losses',
    )
    parser.add_argument('--isBias', action='store_true', default=False)

    # random seed
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

    parser.add_argument('--emb_path', type=str, default="pretrained_emb_dict_SRRSC.pkl")

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

    def load_data(graph):
        ft_dict = dict([(k, graph.nodes[k].data['feature']) for k in graph.ntypes])

        # adj_dict 邻接矩阵
        adj_dict = dict([(k, {}) for k in graph.ntypes])

        # 用于连接预测的负采样
        pos_sample = dict([(k, {}) for k in graph.ntypes])
        neg_sample = dict([(k, {}) for k in graph.ntypes])

        # nt_rel 关系模式
        nt_rel = defaultdict(list)

        for k in adj_dict:
            for etype in graph.canonical_etypes:
                if etype[0] == k:
                    adj_dict[k][etype[2]] = graph.adj(etype=etype)

                    pos_sample[k][etype[2]] = adj_dict[k][etype[2]]._indices()

                    # generate a random permutation of indices from 0 to pos_num - 1
                    pos_num = pos_sample[k][etype[2]].shape[1]
                    perm = torch.randperm(pos_num)
                    # select the first half of the permutation
                    half = perm[: (pos_num // 4) * 3]
                    # select the corresponding edges from pos_sample
                    pos_sample[k][etype[2]] = torch.index_select(
                        pos_sample[k][etype[2]], dim=1, index=half
                    )

                    # 这里为什么正负样本采样数量不一样？
                    neg_sample[k][
                        etype[2]
                    ] = dgl.sampling.global_uniform_negative_sampling(
                        graph,
                        num_samples=pos_sample[k][etype[2]].shape[1],
                        exclude_self_loops=True,
                        etype=etype,
                        replace=False,
                    )
                    rel = str(etype[0]) + '-' + str(etype[2])
                    nt_rel[k].append(rel)

        return ft_dict, adj_dict, nt_rel, pos_sample, neg_sample

    ft_dict, adj_dict, nt_rel, pos_sample, neg_sample = load_data(graph)

    def train():
        """
        start training!
        """
        for k in ft_dict:
            ft_dict[k] = ft_dict[k].to(device)
        for k in adj_dict:
            for kk in adj_dict[k]:
                adj_dict[k][kk] = adj_dict[k][kk].to(device)

        model = modeler(args, nt_rel).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        best = 1e9
        loss_list = []

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            embs = model(adj_dict, ft_dict)
            loss = model.loss_total(embs, ft_dict, adj_dict, pos_sample, neg_sample)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            loss_list.append(train_loss)

            if train_loss < best:
                best = train_loss
                print("Epoch {}, loss {:.5}".format(epoch, train_loss))

        # outs = embs.detach().cpu().numpy()
        # outs = embs

        outs = dict()
        for k, v in embs.items():
            outs[k] = v.detach().cpu().numpy()

        # plot
        # plot(loss_list, args.epochs, args.use_linkpred_emb, args.use_SRRSC_emb)

        return outs

    emb = train()

    # save emb
    save_emb(
        emb,
        idx_node_map,
        idx_node_id_map,
        args.epochs,
        args.use_linkpred_emb,
        args.use_SRRSC_emb,
        args.emb_path,
    )


def save_emb(
    emb, idx_node_map, idx_node_id_map, epoch, use_linkpred_emb, use_SRRSC_emb, emb_name
):
    node_feature_dict = {}

    for ntype in emb.keys():
        node_feature_dict[ntype] = {}
        for i in range(emb[ntype].shape[0]):
            node_name = idx_node_id_map[ntype][i]
            # node_feature_dict[ntype][node_name] = emb[ntype][i].cpu().tolist()
            node_feature_dict[ntype][node_name] = emb[ntype][i]

    # if use_linkpred_emb:
    #     emb_name = "pretrained_emb_dict_linkpred_SRRSC_" + str(epoch) + '.pkl'
    # elif use_SRRSC_emb:
    #     emb_name = "pretrained_emb_dict_SRRSC_" + str(epoch + 10000) + '.pkl'
    # else:
    #     emb_name = "pretrained_emb_dict_SRRSC_" + str(epoch) + '.pkl'

    emb_path = os.path.join('pretrain_emb', emb_name)
    with open(emb_path, 'wb') as f:
        pkl.dump(node_feature_dict, f, pkl.HIGHEST_PROTOCOL)


def plot(train_loss, epoch, use_linkpred_emb, use_SRRSC_emb):
    # 创建画布
    plt.figure()
    plt.plot(
        np.arange(len(train_loss)), np.array(train_loss), c="blue", label="train loss"
    )
    plt.xlabel("train epochs", fontsize=13)
    plt.ylabel("train loss", fontsize=13)
    plt.legend()

    if use_linkpred_emb:
        fig_name = "results/linkpred_SRRSC_train_" + str(epoch) + '.png'
    elif use_SRRSC_emb:
        fig_name = "results/SRRSC_train_" + str(epoch + 10000) + '.pkl'
    else:
        fig_name = "results/SRRSC_train_" + str(epoch) + '.png'

    plt.savefig(fig_name)


if __name__ == "__main__":
    main()
