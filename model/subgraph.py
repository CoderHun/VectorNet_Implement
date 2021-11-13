import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn

def max_pooling(x):
    if x.shape[0] == 0: return x
    else: return torch.max(x, dim=0)[0]

class SubGraph(torch.nn.Module):
    def __init__(self, device, in_feature=6, num_subgraph_layers=3, hidden_feature=64):
        super(SubGraph, self).__init__()
        self.GCN1 = GCN(in_feature, hidden_feature, device)
        self.GCN2 = GCN(in_feature * 2, hidden_feature, device)
        self.GCN3 = GCN(in_feature * 4, hidden_feature, device)
        
    def forward(self, tup):
        x_ = tup[0]
        edge_ = tup[1]
        x, edge_index = self.GCN1(x_, edge_)
        x, edge_index = self.GCN2(x, edge_index)
        x, edge_index = self.GCN3(x, edge_index)
        num_node = x.shape[0]
        node_id = torch.arange(num_node, dtype=torch.int64)
        cluster = torch.ones(num_node, dtype=torch.long)
        cluster = torch.vstack((node_id, cluster))
        out_data = max_pooling(x)
        return out_data

class GCN(torch.nn.Module):
    def __init__(self, in_feature, hidden_feature, device):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feature, hidden_feature)
        self.linear1 = nn.Linear(hidden_feature, in_feature*2)
        self.NormLayer = nn.LayerNorm(hidden_feature)

    def forward(self, x_, edge_):
        x, edge_index = x_, edge_
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.NormLayer(x)
        x = self.linear1(x)
        x = F.relu(x)
        return x, edge_index