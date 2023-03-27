import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *


class modeler(nn.Module):
    def __init__(self, args, nt_rel):
        super().__init__()
        self.args = args
        self.marginloss = nn.MarginRankingLoss(self.args.margin)
        self.b_xent = nn.BCEWithLogitsLoss()
        self.bnn = nn.ModuleDict()
        self.disc2 = nn.ModuleDict()
        self.fc = nn.ModuleDict()
        # gcn 效果不好，不再使用
        self.mygcn = nn.ModuleDict()
        # self.fc1 = nn.Linear(args.emb_dim*2, args.emb_dim)
        # self.fc2 = nn.Linear(args.emb_dim, 1)

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

            # # 在这里增加一个 MyGCN 编码
            # self.mygcn[t] = MyGCN(args.emb_dim, args.hid_units, isBias=args.isBias)

    def forward(self, adj_dict, ft_dict):

        # 虽然是异构图，但是没有用字典形式，而是记录下了不同类别的节点的 index
        # 注意：hid_units 和 out_ft 大小保持一致
        # 和之前的实现保持一致，还是将 emb 写成 dict 的形式
        embs1 = copy.deepcopy(ft_dict)
        embs2 = copy.deepcopy(ft_dict)

        for n, rels in self.nt_rel.items():  # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            for j, rel in enumerate(rels):
                t = rel.split('-')[1]

                # (4328, 256)
                mean_neighbor = torch.spmm(adj_dict[n][t], ft_dict[t])
                v = self.bnn['0' + rel](mean_neighbor)
                # (4328, 256)
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

        # 用学习到的 emb 再来一遍，实现 2-hop 的编码
        for n, rels in self.nt_rel.items():  # p: [[Nei(p-a), Nei(p-c)]]  (Np, Nprel)
            vec = []
            for j, rel in enumerate(rels):
                t = rel.split('-')[-1]

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

            ## 实验效果不好
            ## 1. 如果只是单独进行 SRRSC 的预训练，没有必要把 summary 和 原始特征拼接起来，因为原始特征是随机初始化的，没有意义
            ## 2. 如果是先用 link pred 预训练，然后再用 SRRSC，倒是可以这么操作，看看会不会对结果更好一点
            if self.args.use_linkpred_emb:
                v_cat = torch.hstack((v_summary, ft_dict[n]))
                v_summary = self.fc[n](v_cat)

            embs2[n] = v_summary

        return embs2

    def loss2(self, embs2, ft_dict, adj_dict):

        totalLoss = 0.0

        coef = self.args.lamb

        for n, rels in self.nt_rel.items():

            # nb = len(self.args.node_cnt[n])
            nb = len(ft_dict[n])
            shuf_index = torch.randperm(nb).to(self.device)

            vec = embs2[n]
            fvec = vec[shuf_index]

            # Intrinsic Contrast: 学习的节点嵌入和原始属性的关系，暂时先不用
            # 后续进一步优化的时候，如果采用 link pred 的预训练向量进行初始化，倒是可以考虑加上这部分损失
            # 但是如果不进行联合训练，只使用单个任务，随机初始化的向量是没有意义的

            ones = torch.ones(nb).to(self.device)

            zeros = torch.zeros(nb).to(self.device)
            lbl = torch.cat((ones, zeros), 0).squeeze()
            a = nn.Softmax()(ft_dict[n])
            logits_pos = self.disc2[n](a, vec)
            logits_neg = self.disc2[n](a, fvec)
            logits = torch.hstack((logits_pos, logits_neg))

            totalLoss += 1.0 * self.b_xent(logits, lbl)

            for j, rel in enumerate(rels):
                t = rel.split('-')[-1]

                # 这里直接求 mean 换成 GCN 试试
                mean_neighbor = torch.spmm(adj_dict[n][t], embs2[t])

                # 使用两个 GCN？ 比 mean 应该更科学一点？试试
                # 效果更差了，绝望
                # mean_neighbor = self.mygcn[t](adj_dict[n][t], embs2[t])


                # *：逐元素相乘（对应位置），类似于 torch.mul()
                # 回忆一般的矩阵乘法：torch.mm(), torch.bmm(), torch.matmul()
                logits_pos = (vec * mean_neighbor).sum(-1).view(-1)
                logits_neg = (fvec * mean_neighbor).sum(-1).view(-1)

                # 注意这个 marginloss 函数
                # 前四个参数为 input1, input2, target, margin=self.margin
                # input1 和 input2 是给定的待排序的两个输入，target 代表真实的标签，当 target=1 时，input1 应该排在 input2 前面
                # 计算公式为 loss = max(0, -target * (input1 - input2) + margin)
                totalLoss += coef * self.marginloss(
                    torch.sigmoid(logits_pos), torch.sigmoid(logits_neg), ones
                )

                # 2-hop proximity
                # logits = []
                for k, nr in enumerate(self.nt_rel[t]):
                    tt = nr.split('-')[-1]

                    # nmn = torch.spmm(graph[t][k], embs2[self.args.node_cnt[tt]])
                    # nmn = torch.spmm(graph[n][j], nmn)
                    # 这里其实就不用改了（GCN），因为 coef = 1 loss 的计算用不到以下的代码
                    nmn = torch.spmm(adj_dict[t][tt], embs2[tt])
                    nmn = torch.spmm(adj_dict[n][t], nmn)

                    # 对应位置相乘
                    logits_pos = (vec * nmn).sum(-1).view(-1)
                    logits_neg = (fvec * nmn).sum(-1).view(-1)
                    totalLoss += (1 - coef) * self.marginloss(
                        torch.sigmoid(logits_pos), torch.sigmoid(logits_neg), ones
                    )

        return totalLoss

    def loss_total(self, embs2, ft_dict, adj_dict, pos_sample, neg_sample):
        """
        链接预测和子图预测联合训练时的损失函数，需要同时考虑两部分的损失
        neg_sample: 负采样的边，是一个字典嵌套的形式
        """
        totalLoss = 0.0

        coef = self.args.lamb
        criterion = nn.BCELoss()

        for n, rels in self.nt_rel.items():

            # nb = len(self.args.node_cnt[n])
            nb = len(ft_dict[n])
            shuf_index = torch.randperm(nb).to(self.device)

            vec = embs2[n]
            fvec = vec[shuf_index]

            #### 第一次实验的时候先不考虑这部分损失
            ## Intrinsic Contrast: 学习的节点嵌入和原始属性的关系，暂时先不用
            ## 后续进一步优化的时候，如果采用 link pred 的预训练向量进行初始化，倒是可以考虑加上这部分损失
            ## 但是如果不进行联合训练，只使用单个任务，随机初始化的向量是没有意义的

            ones = torch.ones(nb).to(self.device)

            # zeros = torch.zeros(nb).to(self.device)
            # lbl = torch.cat((ones, zeros), 0).squeeze()
            # a = nn.Softmax()(ft_dict[n])
            # logits_pos = self.disc2[n](a, vec)
            # logits_neg = self.disc2[n](a, fvec)
            # logits = torch.hstack((logits_pos, logits_neg))

            # totalLoss += 1.0 * self.b_xent(logits, lbl)


            ## 第一个损失是子图预测的损失：
            for j, rel in enumerate(rels):
                t = rel.split('-')[-1]

                # 这里直接求 mean 换成 GCN 试试
                mean_neighbor = torch.spmm(adj_dict[n][t], embs2[t])

                # 使用两个 GCN？ 比 mean 应该更科学一点？试试
                # 效果更差了
                # mean_neighbor = self.mygcn[t](adj_dict[n][t], embs2[t])


                # *：逐元素相乘（对应位置），类似于 torch.mul()
                # 回忆一般的矩阵乘法：torch.mm(), torch.bmm(), torch.matmul()
                logits_pos = (vec * mean_neighbor).sum(-1).view(-1)
                logits_neg = (fvec * mean_neighbor).sum(-1).view(-1)

                # 注意这个 marginloss 函数
                # 前四个参数为 input1, input2, target, margin=self.margin
                # input1 和 input2 是给定的待排序的两个输入，target 代表真实的标签，当 target=1 时，input1 应该排在 input2 前面
                # 计算公式为 loss = max(0, -target * (input1 - input2) + margin)
                totalLoss += coef * self.marginloss(
                    torch.sigmoid(logits_pos), torch.sigmoid(logits_neg), ones
                )

                # 2-hop proximity
                # logits = []
                for k, nr in enumerate(self.nt_rel[t]):
                    tt = nr.split('-')[-1]

                    # nmn = torch.spmm(graph[t][k], embs2[self.args.node_cnt[tt]])
                    # nmn = torch.spmm(graph[n][j], nmn)
                    # 这里其实就不用改了（GCN），因为 coef = 1 loss 的计算用不到以下的代码
                    nmn = torch.spmm(adj_dict[t][tt], embs2[tt])
                    nmn = torch.spmm(adj_dict[n][t], nmn)

                    # 对应位置相乘
                    logits_pos = (vec * nmn).sum(-1).view(-1)
                    logits_neg = (fvec * nmn).sum(-1).view(-1)
                    totalLoss += (1 - coef) * self.marginloss(
                        torch.sigmoid(logits_pos), torch.sigmoid(logits_neg), ones
                    )

                ## 第二个损失是链接预测的损失
                # 输入：
                #   embs2：每个节点的向量表示
                #   adj_dict：邻接矩阵
                # 头节点是 n 类型，尾节点是 t 类型，邻接矩阵是 adj_dict[n][t]

                pos_indices = pos_sample[n][t]
                neg_indices = neg_sample[n][t]
                labels = torch.cat([torch.ones(len(pos_indices[0])), torch.zeros(len(neg_indices[0]))]).to(self.device)

                # 不能用全连接层，太大了，超出显存
                # link_pos = self.fc2(F.relu(self.fc1(torch.cat([embs2[n][pos_indices[0]], embs2[t][pos_indices[1]]], dim=1))))
                # link_neg = self.fc2(F.relu(self.fc1(torch.cat([embs2[n][neg_indices[0]], embs2[t][neg_indices[1]]], dim=1))))

                # 直接点积？还是超出内存了，试试采样
                link_pos = (embs2[n][pos_indices[0]] * embs2[t][pos_indices[1]]).sum(-1).view(-1)
                link_neg = (embs2[n][neg_indices[0]] * embs2[t][neg_indices[1]]).sum(-1).view(-1)



                outputs = torch.cat([torch.sigmoid(link_pos), torch.sigmoid(link_neg)], dim=0)
                link_loss = criterion(outputs, labels)
                totalLoss += link_loss



        return totalLoss