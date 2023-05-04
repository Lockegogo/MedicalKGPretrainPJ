import torch
from torch import nn


class MyGCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout=0.5, isBias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(nfeat, nhid))
        nn.init.xavier_uniform_(self.weight)
        if isBias:
            self.bias = nn.Parameter(torch.empty(nhid))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        self.dropout = dropout
        self.act = nn.ReLU()

    def forward(self, adj, x):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)
