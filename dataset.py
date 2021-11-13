#%%
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
#from torch.utils.data import DataLoader
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from tqdm import tqdm
from argoverse.map_representation.map_api import ArgoverseMap
from collections import defaultdict
from torch_geometric.data import InMemoryDataset
from utils.util import *
import os
import copy 
class GraphDataset(Dataset):
    def __init__(self, root_dir, device, verbose=True):
        self.avm =  ArgoverseMap()
        if verbose == True:
            print("Prepare ArgoverseMap()")
        self.afl = ArgoverseForecastingLoader(root_dir)
        if verbose == True:
            print("Prepare ArgoverseForecastingLoader")
        self.train_file_list = os.listdir(root_dir)
        self.train_file_len = len(self.train_file_list)
        print(f"Number of Scene : {self.train_file_len}")
        self.file_pointer = 0
        self.file_list = os.listdir(root_dir)
        self.device = device
        self.root_dir = root_dir
        self.AGENT_ATTR = 0
        self.AGENT_ID = 0
        self.ACTOR_ATTR = 1
        self.LANE_ATTR = 2
        self.POLY_ID = 0
        self.epoch = 0
        self.cache = []

    def update_epoch(self,epochs):
        self.epoch = epochs

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        return_list = []
        for index in range(idx.stop):
            file = self.file_list[index]
            scene = get_trainable_set(self.root_dir, index, file, self.afl, self.avm)

            transforms(scene)
            connect_lane(scene)
            stitching_lane(scene)
            self.POLY_ID = make_agent_fm(scene,self.POLY_ID, 20)
            self.POLY_ID = make_actor_fm(scene, self.POLY_ID, 20)
            self.POLY_ID = make_lane_fm(scene, self.POLY_ID)
            make_edge_index(scene)
            self.POLY_ID = 0

            list_of_data = []
            agent_fm = torch.tensor(scene['agent_fm'], dtype=torch.float)
            agent_edge_matrix = torch.tensor(scene['agent_edge_matrix'], dtype=torch.int64)
            agent_data = Data(x=agent_fm, edge_index=agent_edge_matrix)
            list_of_data.append(agent_data)

            for i, array in enumerate(scene['actor_fm']):
                actor_fm = torch.tensor(array, dtype=torch.float)
                actor_edge_matrix = torch.tensor(scene['actor_edge_matrix'][i], dtype=torch.int64)

                actor_data = Data(x=actor_fm, edge_index=actor_edge_matrix)
                list_of_data.append(actor_data)
            for i, array in enumerate(scene['lane_fm']):
                lane_fm = torch.tensor(array, dtype=torch.float)
                lane_edge_matrix = torch.tensor(scene['lane_edge_matrix'][i], dtype=torch.int64)

                lane_data = Data(x=lane_fm, edge_index=lane_edge_matrix)
                list_of_data.append(lane_data)

            x_gt = torch.tensor(scene['agent']['x'][20:].tolist())
            y_gt = torch.tensor(scene['agent']['y'][20:].tolist())
            Ground_Truth = torch.hstack((x_gt, y_gt))
            return_list.append([list_of_data, Ground_Truth])

        return return_list


    def get_scene(self, idx):
        file = self.file_list[idx]
        scene = get_trainable_set(self.root_dir, idx, file, self.afl, self.avm)
        transforms(scene)
        connect_lane(scene)
        stitching_lane(scene)
        return scene

class GraphDataset2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, verbose=False):
        super(GraphDataset2, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['dataset.pt']
    def download(self):
        pass
    def process(self):
        verbose = True
        file_len = 200
        root = '/media/mmc-server1/Server2/hun/Argoverse/data/train/train/data'
        self.avm =  ArgoverseMap()
        if verbose == True:
            print("Prepare ArgoverseMap()")
        self.afl = ArgoverseForecastingLoader(root)
        if verbose == True:
            print("Prepare ArgoverseForecastingLoader")
        self.train_file_list = os.listdir(root)
        self.train_file_len = len(self.train_file_list)
        print(f"Number of Scene : {self.train_file_len}")
        self.file_pointer = 0
        self.file_list = os.listdir(root)
        self.file_len = file_len
        self.root_dir = root
        self.AGENT_ATTR = 0
        self.AGENT_ID = 0
        self.ACTOR_ATTR = 1
        self.LANE_ATTR = 2
        self.POLY_ID = 0
        self.epoch = 0
        self.cache = []
        # make sure deterministic results
        data_ls = []
        for index in range(self.file_len):
            file = self.file_list[index]
            scene = get_trainable_set(self.root_dir, index, file, self.afl, self.avm)

            transforms(scene)
            connect_lane(scene)
            stitching_lane(scene)
            self.POLY_ID = make_agent_fm(scene,self.POLY_ID, 20)
            self.POLY_ID = make_actor_fm(scene, self.POLY_ID, 20)
            self.POLY_ID = make_lane_fm(scene, self.POLY_ID)
            make_edge_index(scene)
            self.POLY_ID = 0

            list_of_data = []
            agent_fm = torch.tensor(scene['agent_fm'], dtype=torch.float)
            agent_edge_matrix = torch.tensor(scene['agent_edge_matrix'], dtype=torch.int64)
            x_gt = torch.tensor(scene['agent']['x'][20:].tolist())
            y_gt = torch.tensor(scene['agent']['y'][20:].tolist())
            Ground_Truth = torch.hstack((x_gt, y_gt))
            agent_data = Data(x=agent_fm, y=Ground_Truth, edge_index=agent_edge_matrix)
            data_ls.append(agent_data)

            for i, array in enumerate(scene['actor_fm']):
                actor_fm = torch.tensor(array, dtype=torch.float)
                actor_edge_matrix = torch.tensor(scene['actor_edge_matrix'][i], dtype=torch.int64)

                actor_data = Data(x=actor_fm, edge_index=actor_edge_matrix)
                data_ls.append(actor_data)
            for i, array in enumerate(scene['lane_fm']):
                lane_fm = torch.tensor(array, dtype=torch.float)
                lane_edge_matrix = torch.tensor(scene['lane_edge_matrix'][i], dtype=torch.int64)

                lane_data = Data(x=lane_fm, edge_index=lane_edge_matrix)
                data_ls.append(lane_data)
        data, slices = self.collate(data_ls)
        torch.save((data, slices), self.processed_paths[0])
        return data_ls
def get_Data():
    verbose = True
    file_len = 200
    root = '/media/mmc-server1/Server2/hun/Argoverse/data/train/train/data'
    avm =  ArgoverseMap()
    if verbose == True:
        print("Prepare ArgoverseMap()")
    afl = ArgoverseForecastingLoader(root)
    if verbose == True:
        print("Prepare ArgoverseForecastingLoader")
    train_file_list = os.listdir(root)
    train_file_len = len(train_file_list)
    print(f"Number of Scene : {train_file_len}")
    file_pointer = 0
    file_list = os.listdir(root)
    file_len = file_len
    root_dir = root
    AGENT_ATTR = 0
    AGENT_ID = 0
    ACTOR_ATTR = 1
    LANE_ATTR = 2
    POLY_ID = 0
    epoch = 0
    cache = []
    # make sure deterministic results
    data_ls = []

    for index in range(file_len):
        file = file_list[index]
        scene = get_trainable_set(root_dir, index, file, afl, avm)

        transforms(scene)
        connect_lane(scene)
        stitching_lane(scene)
        POLY_ID = make_agent_fm(scene,POLY_ID, 20)
        POLY_ID = make_actor_fm(scene, POLY_ID, 20)
        POLY_ID = make_lane_fm(scene, POLY_ID)
        make_edge_index(scene)
        POLY_ID = 0

        list_of_data = []
        agent_fm = torch.tensor(scene['agent_fm'], dtype=torch.float)
        agent_edge_matrix = torch.tensor(scene['agent_edge_matrix'], dtype=torch.int64)
        x_gt = torch.tensor(scene['agent']['x'][20:].tolist())
        y_gt = torch.tensor(scene['agent']['y'][20:].tolist())
        Ground_Truth = torch.hstack((x_gt, y_gt))
        agent_data = Data(x=agent_fm, y=Ground_Truth, edge_index=agent_edge_matrix)
        list_of_data.append(agent_data)

        for i, array in enumerate(scene['actor_fm']):
            actor_fm = torch.tensor(array, dtype=torch.float)
            actor_edge_matrix = torch.tensor(scene['actor_edge_matrix'][i], dtype=torch.int64)

            actor_data = Data(x=actor_fm, edge_index=actor_edge_matrix)
            list_of_data.append(actor_data)
        for i, array in enumerate(scene['lane_fm']):
            lane_fm = torch.tensor(array, dtype=torch.float)
            lane_edge_matrix = torch.tensor(scene['lane_edge_matrix'][i], dtype=torch.int64)

            lane_data = Data(x=lane_fm, edge_index=lane_edge_matrix)
            list_of_data.append(lane_data)
        data_ls.append(list_of_data)
    return data_ls

#%%
if __name__ == "__main__":
    from torch.nn.loader import DataListLoader
    from model.VectorNet import VectorNet
    from torch.nn import DataParallel
    import torch.optim as optim
    gpus = [0, 1, 2, 3]
    gpu_ids = ['cuda:0','cuda:1','cuda:2','cuda:3']

    decay_lr_factor = 0.3
    decay_lr_every = 5
    lr = 0.001

    USE_CUDA = torch.cuda.is_available()
    print("CUDA USE : {}".format(USE_CUDA))

    device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')

    model = VectorNet(sub_in_feature=6, output_horizon = 60 , device=device, sub_out_feature=64, num_subgraph_layers=3, num_global_graph_layer=1,
                global_out_feature=64, hidden_nodes=64)
    #model = DataParallel(model, device_ids=gpu_ids, output_device=gpus[0]).to(device)
    dataset = get_Data()
    dataloader = DataListLoader(dataset, batch_size=32)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)

    model.train()
    for item in dataloader:
        # batch unlock
        print(f"item type : {type(item)}")
        for i in item:
            print(f"i type : {type(i)}")
            out = model(i) # item == list(Data)
            print(out.shape)
            break
        break
        