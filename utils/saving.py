#%%
import os
import torch
from utils.visualizer import viz_sequence_rasterize
from utils.visualizer import draw_test_fig
import numpy as np

def save_checkpoint(checkpoint_dir, model, optimizer, end_epoch, val_minade, date):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'end_epoch' : end_epoch,
        'val_minade': val_minade
        }
    checkpoint_path = checkpoint_dir + f'/Epoch_{end_epoch}.MinADE_{val_minade:.1f}.pth'
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)
    print('model MinADE %s' % state['val_minade'])
    return state['end_epoch']

# Not Used
def save_img(scene, dataset, Ground_Truth, horizon, dir):
    df = scene['df']
    agent = scene['agent']
    avm = dataset.avm
    horizon = int(horizon/2)
    x = np.array(Ground_Truth[:30], dtype=np.float64).tolist()
    y = np.array(Ground_Truth[30:], dtype=np.float64).tolist()
    gt = (x, y)
    img = viz_sequence_rasterize(scene, agent, df, avm, gt, dir)
#%%
# Debugging Code
if __name__ == "__main__":
    from dataset import GraphDataset
    SAVE_DIR = '/media/mmc-server1/Server2/hun/Paper_Implement/VectorNet/my_implement/checkpoint'
    SAMPLE_DIR = "/media/mmc-server1/Server2/hun/Argoverse/forecasting_sample/data"
    device = torch.device('cpu')
    train_dataset = GraphDataset(SAMPLE_DIR, device)
    scene = train_dataset.train_set['0']
    gt_x = torch.linspace(0,10, steps=30)
    gt_y = torch.linspace(0,10, steps=30)
    gt = torch.hstack((gt_x,gt_y))

    pred_x = torch.linspace(0,10, steps=30) * 1.25
    pred_y = torch.linspace(0,10, steps=30) * 1.5
    pred = torch.hstack((pred_x,pred_y))
    draw_test_fig(train_dataset, scene, gt, pred, SAVE_DIR)