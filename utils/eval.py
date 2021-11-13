import torch
import numpy as np
import random

def get_MinADE(model, val_dataloader, horizon, batch_size):
    model.eval()
    with torch.no_grad():
        for data_batch in val_dataloader:
            batch_data = []
            Ground_Truth = None
            for i, data in enumerate(data_batch):
                if len(data[0]) < 2: continue 
                batch_data.append(data[0])
                if i == 0: Ground_Truth = data[1]
                else: Ground_Truth = torch.vstack((Ground_Truth, data[1]))
            Ground_Truth = Ground_Truth.cpu()
            out = model(batch_data).cpu()
            minade = MinADE(Ground_Truth, out, int(horizon/2))
            model.train()
            random_num = random.randrange(0,batch_size)
            # Mini Validate
            return minade, random_num, Ground_Truth[random_num], out[random_num]

def MinADE(gt, pred, horizon):
    size = pred.shape[0]
    ade_list = []
    for idx in range(size):
        x1 = gt[idx][:horizon]
        y1 = gt[idx][horizon:]
        x2 = pred[idx][:horizon]
        y2 = pred[idx][horizon:]
        ade_list.append(torch.sqrt(torch.sum((x1 - x2)**2 + (y1 - y2)**2)))
    return np.min(ade_list)

if __name__ == "__main__":
    one = torch.ones(5,60)
    two = torch.ones(5,60) + torch.ones(5,60)
    horizon = 30
    a = MinADE(one, two, horizon)
    print(a)