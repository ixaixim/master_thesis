# %%
import os
import torch
import numpy as np
import pandas as pd
import h5py
from random import shuffle
import dill

import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoLoader
from dimenet_custom_model import DimeNet, DimeNet_custom

from ase.data import atomic_numbers
from train import train, test, EarlyStopper

########################################################################
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'DEVICE used: {DEVICE}')
# hyperparams
lr = 0.01
EPOCHS = 2000
BATCHSIZE = 16
VAL_BATCHSIZE = 8
dataset_len = 2000
train_len = 1500
val_len = 250
test_len = dataset_len - (train_len+val_len)
# debug
debug = False
if debug == True:
    lr = 0.001
    EPOCHS = 10
    BATCHSIZE = 10
    VAL_BATCHSIZE = 4
    dataset_len = 100
    train_len = 75
    val_len = 20
    test_len = dataset_len - (train_len+val_len)

########################################################################

with h5py.File("data/raw/john_pentacene.h5", "r") as file:
    print(f'Extracting data from file')
    # get content and convert to numpy arrays
    atoms = file.get("atoms")[()].astype(str)
    positions = file.get("positions")[()]
    transfer_integrals = file.get("transfer_integrals")[()]

# apply log
mask = ~(transfer_integrals==0.)
transfer_integrals = np.log(np.abs(transfer_integrals))[mask]
positions = positions[mask]

atoms = torch.LongTensor([atomic_numbers[s] for s in atoms])
positions = torch.Tensor(positions)
transfer_integrals = torch.Tensor(transfer_integrals)


dataset = []
for i in range(dataset_len):
    dataset.append(Data(pos=positions[i], y=[0,transfer_integrals[i]]))
train_set = dataset[:train_len]
val_set = dataset[train_len:train_len+val_len]
test_set = dataset[train_len+val_len:]
train_loader = GeoLoader(train_set, shuffle=True, batch_size=BATCHSIZE)
val_loader = GeoLoader(val_set, shuffle=True, batch_size=VAL_BATCHSIZE)
test_loader = GeoLoader(test_set, shuffle=True, batch_size=VAL_BATCHSIZE)

########################################################################
if debug==False:
    model = DimeNet(hidden_channels=64,
                    out_channels=1,
                    num_blocks=6,
                    num_bilinear=8,
                    num_spherical=7,
                    num_radial=6,
                    cutoff=1.5)
# debug
else: 
    model = DimeNet(
                    hidden_channels=2,
                    out_channels=1,
                    num_blocks=2,
                    num_bilinear=8,
                    num_spherical=3,
                    num_radial=1,
                    )
    
model = model.to(DEVICE)
########################################################################
# debug
# pass a batch of molecules (test valid only for pentacene)
# batch = next(iter(train_loader))
# alpha = 2.0
# model(z=atoms.repeat(BATCHSIZE), pos=positions[:BATCHSIZE].view(-1,3), alpha=alpha, batch = batch.batch)

# pass one molecule 
# model(z=atoms, pos=positions[0].view(-1,3))
########################################################################

optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
criterion = F.mse_loss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=10,
                                                        min_lr=0.0000001)

loss_history = []
early_stopper = EarlyStopper(patience=70, min_delta=0.1)
for epoch in range(1, EPOCHS+1):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch, DEVICE, model, criterion, optimizer, train_loader)
    val_loss, val_error = test(DEVICE, model, criterion, val_loader)
    scheduler.step(val_error)

    if early_stopper.early_stop(val_loss):  
        break
    # TODO: add a condition to save model: 
    # if early_stopper.counter == 0:
        # model save

    print(  f'Epoch: {epoch:03d},' 
            f'LR: {lr:.7f},'
            f'train_loss: {loss:.7f},'
            f'MAE: {val_error:.7f}')
    loss_history.append({"train":loss, "MAE":val_error}) 

########################################################################

# save model 
model_path = os.getcwd() + f'/dimenet/weights/model_{epoch}.pth'
# torch.save(model, model_path)           
with open(model_path, 'wb') as f:
    dill.dump(model, f)

# save test_set

data_path = os.getcwd() + f'/dimenet/data/data_{i}.pt'
for i in range(test_len):
    torch.save(test_set[i], data_path)