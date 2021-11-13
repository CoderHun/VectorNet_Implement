#%%
from utils.saving import save_checkpoint, load_checkpoint
from dataset import GraphDataset
from torch_geometric.loader import DataListLoader
from utils.eval import get_MinADE
from utils. visualizer import draw_simple_fig
from model.VectorNet import VectorNet
import time
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm
import os
#%%
# ! Made by : Hun ! #
### Functions ###
def get_train_batch(data_batch):
    batch_Data = []
    Ground_Truth = None
    for i, data in enumerate(data_batch):
        if len(data[0]) < 2:
            continue 
        batch_Data.append(data[0])
        if i == 0 or Ground_Truth is None: Ground_Truth = data[1]
        else: Ground_Truth = torch.vstack((Ground_Truth, data[1]))
    return batch_Data, Ground_Truth

def validation(epoch, best_minade):
    curr_minade, index, gt, pred = get_MinADE(model, val_loader, horizon, batch_size)
    print(f"eval as epoch:{epoch}     validation minADE : {curr_minade}")
    name = SAVE_DIR + f"/epoch_{epoch}_MinADE_{curr_minade}.png"
    draw_simple_fig(pred, gt, name)
    save_checkpoint(SAVE_DIR, model, optimizer, epoch, best_minade, date)
    if curr_minade < best_minade:
        best_minade = curr_minade
    return best_minade

def display(global_step):
    if (global_step + 1) % show_every == 0:
        print( f"loss at epoch {epoch} step {batch_index*batch_size/dataset_size * 100:3f}%   :  {loss.item():3f}, \
        lr:{optimizer.state_dict()['param_groups'][0]['lr']: .6f}, time:{time.time() - start_tic: 4f}sec")

### Directory ###
SAMPLE_DIR = "Argoverse Sample Dir"
TRAIN_DIR = 'Argoverse Train Dir'
VAL_DIR = 'Argoverse Validation Dir'
SAVE_DIR = 'checkpoint'
MODEL_PATH = 'Your Model Path'
LOAD_MODEL = False

### Training Option ###
epochs = 100
train_dataset_size = 20000
val_dataset_size = 500
batch_size = 500
decay_lr_factor = 0.5
decay_lr_every = 5
lr = 0.001
show_every = 1
best_minade = float('inf')
global_step = 0
date = '21/11/13'

### Evaluation related ###
draw_every = 1
val_every = 3
horizon = 60

### GPU Option ###
gpus = [0]
USE_CUDA = torch.cuda.is_available()
device = torch.device(f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu')
print("CUDA USE : {}".format(USE_CUDA))
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())

### Dataset ###
train_dataset = GraphDataset(TRAIN_DIR, device, verbose=True)
train_loader = DataListLoader(train_dataset[:train_dataset_size], batch_size=batch_size)
val_dataset = GraphDataset(VAL_DIR,device, verbose=False)
val_loader = DataListLoader(val_dataset[:val_dataset_size], batch_size=batch_size)

#%%
### Model and Optimizer ###
model = VectorNet(sub_in_feature=6, output_horizon = horizon , device=device, sub_out_feature=64, num_subgraph_layers=3, num_global_graph_layer=1,
                global_out_feature=64, hidden_nodes=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)
if LOAD_MODEL == True:
    epoch = load_checkpoint(MODEL_PATH, model, optimizer)
model.train()

#%%
### Train Loop ###
for epoch in range(epochs):
    global_step =  0
    start_tic = time.time()
    for batch_index, data_batch in enumerate(train_loader):
        batch_data, Ground_Truth = get_train_batch(data_batch)
        optimizer.zero_grad()
        output = model(batch_data)
        loss = F.mse_loss(output, Ground_Truth.to(device))
        loss.backward()
        optimizer.step()
        global_step += 1
        display(global_step)
    if (epoch+1) % val_every == 0:
        best_minade = validation(epoch, best_minade)
    train_dataset.update_epoch(epoch)
    scheduler.step()