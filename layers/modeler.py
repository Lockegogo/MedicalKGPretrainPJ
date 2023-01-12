import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
import copy


class modeler(nn.Module):
    def __init__(self, args, nt_rel):
        super().__init__()
        self.args = args
        self.marginloss = nn.MarginRankingLoss(self.args.margin)
        self.b_xent = nn.BCEWithLogitsLoss()
        self.bnn = nn.ModuleDict()
        self.disc2 = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        self.nt_rel = nt_rel

        self.semanticatt = nn.ModuleDict()

        self.device = (
            torch.device("cuda:" + str(args.device))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        # nt_rel

        for t, rels in self.nt_rel.items():  # {note_type: [rel1, rel2]}

            self.fc[t] = FullyConnect(
                args.hid_units2 + args.emb_dim,
                args.out_ft,
                drop_prob=self.args.drop_prob,
            )
            self.disc2[t] = Discriminator(args.emb_dim, args.out_ft)

            for rel in rels:
                # 两层
                self.bnn['0' + rel] = GCN(
                    args.emb_dim, args.hid_units, act=nn.ReLU(), isBias=args.isBias
                )
                self.bnn['1' + rel] = GCN(
                    args.hid_units, args.hid_units2, act=nn.ReLU(), isBias=args.isBias
                )

            # 语义编码
            self.semanticatt['0' + t] = SemanticAttention(
                args.hid_units, args.hid_units // 4
            )
            self.semanticatt['1' + t] = SemanticAttention(
                args.hid_units2, args.hid_units2 // 4
            )

    def forward(self, adj_dict, ft_dict):
        totalLoss = 0.0
        reg_loss = 0.0

        # # 虽然是异构图，但是没有用字典形式，而是记录下了不同类别的节点的 index
        # embs1 = torch.zeros((self.args.node_size, self.args.hid_units)).to(
        #     self.device
        # )
        # embs2 = torch.zeros((self.args.node_size, self.args.out_ft)).to(
        #     self.device
        # )
        # # 注意：hid_units 和 out_ft 大小保持一致

        # 和之前的实现保持一致，还是将 emb 写成 dict 的形式
        embs1 = copy.deepcopy(ft_dict)
        embs2 = copy.deepcopy(ft_dict)

        for n, rels in self.nt_rel.items():  # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            for j, rel in enumerate(rels):
                t = rel.split('-')[1]
                # graph[n][j]：第 n 类节点的第 j 个关系的邻接矩阵？
                # mean_neighbor = torch.spmm(graph[n][j], features[self.args.node_cnt[t]])
                mean_neighbor = torch.spmm(adj_dict[n][t], ft_dict[t])

                v = self.bnn['0' + rel](mean_neighbor)
                vec.append(v)  # (1, Nt, ft)

            # (2, 4328, 256)
            vec = torch.stack(vec, 0)  # (rel_size, Nt, emb_dim)

            if self.args.isAtt:
                v_summary = self.semanticatt['0' + n](
                    vec.view(-1, self.args.hid_units), len(rels)
                )
            else:
                # (4328, 256)
                v_summary = torch.mean(vec, 0)  # (Nt, hd)

            # embs1[self.args.node_cnt[n]] = v_summary
            embs1[n] = v_summary

        # 用学习到的 emb 再来一遍
        # 来实现 2-hop 的编码吗？
        for n, rels in self.nt_rel.items():  # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            for j, rel in enumerate(rels):
                t = rel.split('-')[-1]

                # mean_neighbor = torch.spmm(graph[n][j], embs1[self.args.node_cnt[t]])
                mean_neighbor = torch.spmm(adj_dict[n][t], embs1[t])

                v = self.bnn['1' + rel](mean_neighbor)
                vec.append(v)  # (1, Nt, ft)

            vec = torch.stack(vec, 0)  # (rel_size, Nt, emb_dim)
            if self.args.isAtt:
                v_summary = self.semanticatt['1' + n](
                    vec.view(-1, self.args.hid_units2), len(rels)
                )
            else:
                v_summary = torch.mean(vec, 0)  # (Nt, hd)


            ## 这里有点问题
            # 1. 如果只是单独进行 SRRSC 的预训练，没有必要把 summary 和 原始特征拼接起来，因为原始特征是随机初始化的，没有意义
            # 2. 如果是先用 link pred 预训练，然后再用 SRRSC，倒是可以这么操作，看看会不会对结果更好一点
            v_cat = torch.hstack((v_summary, ft_dict[n]))
            v_summary = self.fc[n](v_cat)

            # embs2[self.args.node_cnt[n]] = v_summary
            embs2[n] = v_summary

        return embs2

    def loss2(self, embs2, ft_dict, adj_dict):

        totalLoss = 0.0
        # embs = torch.zeros((self.args.node_size, self.args.out_ft)).to(self.device)
        embs = copy.deepcopy(ft_dict)

        coef = self.args.lamb

        for n, rels in self.nt_rel.items():

            # nb = len(self.args.node_cnt[n])
            nb = len(ft_dict[n])
            ones = torch.ones(nb).to(self.device)
            zeros = torch.zeros(nb).to(self.device)
            lbl = torch.cat((ones, zeros), 0).squeeze()

            shuf_index = torch.randperm(nb).to(self.device)

            # vec = embs2[self.args.node_cnt[n]]
            vec = embs2[n]

            fvec = vec[shuf_index]

            # a = nn.Softmax()(features[self.args.node_cnt[n]])
            a = nn.Softmax()(ft_dict[n])

            logits_pos = self.disc2[n](a, vec)
            logits_neg = self.disc2[n](a, fvec)
            logits = torch.hstack((logits_pos, logits_neg))

            totalLoss += 1.0 * self.b_xent(logits, lbl)

            for j, rel in enumerate(rels):
                t = rel.split('-')[-1]

                # mean_neighbor = torch.spmm(graph[n][j], embs2[self.args.node_cnt[t]])
                mean_neighbor = torch.spmm(adj_dict[n][t], embs2[t])

                logits_pos = (vec * mean_neighbor).sum(-1).view(-1)
                logits_neg = (fvec * mean_neighbor).sum(-1).view(-1)

                totalLoss += coef * self.marginloss(
                    torch.sigmoid(logits_pos), torch.sigmoid(logits_neg), ones
                )

                # 2-hop proximity
                logits = []
                for k, nr in enumerate(self.nt_rel[t]):
                    tt = nr.split('-')[-1]

                    # nmn = torch.spmm(graph[t][k], embs2[self.args.node_cnt[tt]])
                    # nmn = torch.spmm(graph[n][j], nmn)
                    nmn = torch.spmm(adj_dict[t][tt], embs2[tt])
                    nmn = torch.spmm(adj_dict[n][t], nmn)

                    # 对应位置相乘
                    logits_pos = (vec * nmn).sum(-1).view(-1)
                    logits_neg = (fvec * nmn).sum(-1).view(-1)
                    totalLoss += (1 - coef) * self.marginloss(
                        torch.sigmoid(logits_pos), torch.sigmoid(logits_neg), ones
                    )


        return totalLoss
