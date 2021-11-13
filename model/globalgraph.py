import torch
import torch.nn as nn

class GlobalGraph(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super(GlobalGraph, self).__init__()
        self.in_feature = in_feature
        self.q_lin = nn.Linear(48, 64)
        self.k_lin = nn.Linear(48, 64)
        self.v_lin = nn.Linear(48, 64)

    def forward(self, x):
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)

        scores = torch.mm(query.transpose(1, 0), key)
        #print(f"score shape : {scores.shape}")
        attention_weights = nn.functional.softmax(scores, dim=-1)
        #print(f"weight shape: {attention_weights.shape}")
        result = torch.mm(value, attention_weights)
        result = torch.mean(result,dim=0)
        return result