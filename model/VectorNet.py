#%%
from .decoder import Decoder
from .subgraph import SubGraph
from .globalgraph import GlobalGraph
import torch.nn as nn
import torch
from torch_geometric.data import Data
import copy

#%%

class VectorNet(nn.Module):
    def __init__(self, sub_in_feature, output_horizon, device, sub_out_feature=48, num_subgraph_layers=3, num_global_graph_layer=1,
                global_out_feature=64, hidden_nodes=64):
        super(VectorNet, self).__init__()
        self.subgraph = SubGraph(device, sub_in_feature, num_subgraph_layers, sub_out_feature)
        self.GlobalGraph = GlobalGraph(sub_out_feature, global_out_feature)
        self.Decoder = Decoder(global_out_feature, output_horizon, hidden_nodes)
        self.device = device

    def apply_subgraph(self, Data_list):
        subgraph_out_stack = None
        for i, obj in enumerate(Data_list):
            subgraph_out = self.subgraph( (copy.deepcopy(obj.x).to(self.device), copy.deepcopy(obj.edge_index).to(self.device)) )
            if i == 0: subgraph_out_stack = subgraph_out
            else: subgraph_out_stack = torch.vstack((subgraph_out_stack, subgraph_out))
        return subgraph_out_stack

    def forward(self, batch_Data):
        globalgraph_out_stack = None
        for idx, Data_list in enumerate(batch_Data):
            subgraph_out_stack = self.apply_subgraph(Data_list)
            if idx == 0: globalgraph_out_stack = self.GlobalGraph(subgraph_out_stack)
            else:
                out = self.GlobalGraph(subgraph_out_stack)
                globalgraph_out_stack = torch.vstack((globalgraph_out_stack, out))
        pred_traj = self.Decoder(globalgraph_out_stack)
        return pred_traj

#%%
if __name__ == "__main__":
    agent_fm = torch.ones(10,6,dtype=torch.float)
    agent_edge = torch.ones(2,8, dtype=torch.int64)

    actor_fm = torch.ones(10,6,dtype=torch.float)
    actor_edge = torch.ones(2,8, dtype=torch.int64)

    lane_fm = torch.ones(10,6,dtype=torch.float)
    lane_edge = torch.ones(2,8, dtype=torch.int64)

    agent_data = Data(x=agent_fm, edge_index=agent_edge)
    actor_list = []
    for i in range(3):
        actor_data = Data(x=actor_fm, edge_index=actor_edge)
        actor_list.append(actor_data)
    lane_list = []
    for i in range(10):
        lane_data = Data(x=lane_fm, edge_index=lane_edge)
        lane_list.append(lane_data)
    Ground_Truth = torch.ones(1,30,dtype=torch.float)
    set = [agent_data, actor_list, lane_list, Ground_Truth]
    set2 = copy.deepcopy(set)
    sample = [set,set2]
    vectornet = VectorNet(6,60)
    result = vectornet(sample)
    print(f"result shape : {result.shape}")