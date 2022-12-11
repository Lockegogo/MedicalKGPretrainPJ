import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch

## GCN
class GCN(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(in_feat, hidden_feat, norm='right')
                for rel in rel_names
            }
        )
        self.conv2 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(hidden_feat, out_feat, norm='right')
                for rel in rel_names
            }
        )

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = self.conv2(blocks[1], x)
        return x


## GAT
class GAT(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GATConv(in_feat, hidden_feat, num_heads=2)
                for rel in rel_names
            },
            aggregate='sum',
        )
        self.conv2 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GATConv(hidden_feat * 2, out_feat, num_heads=1)
                for rel in rel_names
            },
            aggregate='sum',
        )
        self.hidden_feat = hidden_feat
        self.out_feat = out_feat

    def forward(self, blocks, x):

        x = self.conv1(blocks[0], x)
        x = {k: F.relu(v.reshape(-1, self.hidden_feat * 2)) for k, v in x.items()}
        x = self.conv2(blocks[1], x)
        x = {k: v.reshape(-1, self.out_feat) for k, v in x.items()}
        return x


## GraphSAGE
class GraphSAGE(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.SAGEConv(in_feat, hidden_feat, aggregator_type='mean')
                for rel in rel_names
            },
            aggregate='sum',
        )
        self.conv2 = dglnn.HeteroGraphConv(
            {
                rel: dglnn.SAGEConv(hidden_feat, out_feat, aggregator_type='mean')
                for rel in rel_names
            },
            aggregate='sum',
        )
        self.hidden_feat = hidden_feat
        self.out_feat = out_feat

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = {k: F.relu(v.reshape(-1, self.hidden_feat)) for k, v in x.items()}
        x = self.conv2(blocks[1], x)
        x = {k: v.reshape(-1, self.out_feat) for k, v in x.items()}
        return x


class GNN(nn.Module):
    """
    Output: representations
    """
    def __init__(
        self, in_features, hidden_features, out_features, etypes, gnn_type="GraphSAGE"
    ):
        super(GNN, self).__init__()

        if gnn_type == 'GCN':
            self.gnns = GCN(in_features, hidden_features, out_features, etypes)
        elif gnn_type == 'GAT':
            self.gnns = GAT(in_features, hidden_features, out_features, etypes)
        elif gnn_type == 'GraphSAGE':
            self.gnns = GraphSAGE(in_features, hidden_features, out_features, etypes)

    def forward(self, blocks, x):
        h = self.gnns(blocks, x)
        return h



class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of which is a dictionary representing the features of the source nodes, the destination nodes, and the edges themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['feature'], edges.dst['feature']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['feature'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']




if __name__ == "__main__":
    pass
