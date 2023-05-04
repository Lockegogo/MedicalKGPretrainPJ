import dgl
import numpy as np
import torch.nn as nn


class ScorePredictor(nn.Module):
    def forward(self, subgraph, x):
        with subgraph.local_scope():
            subgraph.ndata['x'] = x
            for etype in subgraph.canonical_etypes:
                subgraph.apply_edges(
                    dgl.function.u_dot_v('x', 'x', 'score'), etype=etype
                )
            return subgraph.edata['score']


# negative sampling
class NegativeSampler(object):
    def __init__(self, g, k):
        self.weights = g.in_degrees().float() ** 0.85
        self.k = k

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        src = src.repeat_interleave(self.k)
        dst = self.weights.multinomial(len(src), replacement=True).long()
        return src, dst


def split_train_test(graph, test_ratio, val_ratio, device):
    """
    edge split for link prediction
    return: 4 subgraph
    TODO: 验证集和测试集负采样的时候是不是不能采到训练集中的正样本？
    """
    negativeSample = NegativeSampler(graph, 1)

    eids = np.arange(graph.number_of_edges())
    eids = np.random.permutation(eids)
    u, v = graph.edges()

    test_size = int(len(eids) * test_ratio)
    test_eids = eids[:test_size]
    test_pos_u, test_pos_v = u[test_eids], v[test_eids]
    test_neg_u, test_neg_v = negativeSample(graph, test_eids)
    test_neg_v = test_neg_v.long()

    val_size = int(len(eids) * val_ratio)
    val_eids = eids[test_size : test_size + val_size]
    val_pos_u, val_pos_v = u[val_eids], v[val_eids]
    val_neg_u, val_neg_v = negativeSample(graph, val_eids)
    val_neg_v = val_neg_v.long()

    train_eids = eids[test_size + val_size:]
    train_pos_u, train_pos_v = u[train_eids], v[train_eids]
    train_neg_u, train_neg_v = negativeSample(graph, train_eids)
    train_neg_v = train_neg_v.long()

    test_pos_graph = node_to_graph(graph, test_pos_u, test_pos_v, device)
    test_neg_graph = node_to_graph(graph, test_neg_u, test_neg_v, device)

    val_pos_graph = node_to_graph(graph, val_pos_u, val_pos_v, device)
    val_neg_graph = node_to_graph(graph, val_neg_u, val_neg_v, device)

    train_pos_graph = node_to_graph(graph, train_pos_u, train_pos_v, device)
    train_neg_graph = node_to_graph(graph, train_neg_u, train_neg_v, device)

    return (
        test_pos_graph,
        test_neg_graph,
        val_pos_graph,
        val_neg_graph,
        train_pos_graph,
        train_neg_graph,
    )


def node_to_graph(graph, u, v, device):
    subgraph = dgl.graph((u, v), num_nodes=graph.number_of_nodes())
    subgraph.ndata['feature'] = graph.ndata['feature']
    subgraph = subgraph.to(device)
    return subgraph


if __name__ == "__main__":
    pass

