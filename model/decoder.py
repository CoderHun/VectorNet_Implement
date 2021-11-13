import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, in_nodes, out_nodes, hidden_nodes):
        super(Decoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_nodes, hidden_nodes),
            nn.LayerNorm(hidden_nodes),
            nn.ReLU(),
            nn.Linear(hidden_nodes, out_nodes)
        )
    def forward(self, x):
        return self.mlp(x)