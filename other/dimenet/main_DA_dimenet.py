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
from train import train_grl, test_grl, EarlyStopper

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'DEVICE used: {DEVICE}')
# hyperparams
lr = 0.01
EPOCHS = 2000
BATCHSIZE = 32
VAL_BATCHSIZE = 16
dataset_len = 2000
train_len = 1500
val_len = 250
test_len = dataset_len - (train_len+val_len)
train_labels = [0] # source domain: pentacene (0) target domain: tetracene (1)
# debug
debug = True
if debug == True:
    lr = 0.001
    EPOCHS = 10
    BATCHSIZE = 10
    VAL_BATCHSIZE = 4
    dataset_len = 100
    # lenghts of dataset sources 
    len_dataset_domain_0 = 80
    len_dataset_domain_1 = dataset_len -len_dataset_domain_0
    len_dataset_domain = {0: len_dataset_domain_0, 1: len_dataset_domain_1 }
    # 
    train_len = 75
    val_len = 20
    test_len = dataset_len - (train_len+val_len)

########################################################################
atoms = {}
positions = {}
transfer_integrals = {}
with h5py.File("data/raw/john_pentacene.h5", "r") as file:
    print(f'Extracting data from file')
    # get content and convert to numpy arrays
    atoms[0] = file.get("atoms")[()].astype(str)
    positions[0] = file.get("positions")[()]
    transfer_integrals[0] = file.get("transfer_integrals")[()]

with h5py.File("data/raw/john_tetracene.h5", "r") as file:
    print(f'Extracting data from file')
    # get content and convert to numpy arrays
    atoms[1] = file.get("atoms")[()].astype(str)
    positions[1] = (file.get("positions")[()])
    transfer_integrals[1] = (file.get("transfer_integrals")[()])

dataset = {key: [] for key in atoms.keys()}
i = 0 
for key in sorted(positions.keys()):
    # apply log
    mask = ~(transfer_integrals[key]==0.)
    transfer_integrals[key] = np.log(np.abs(transfer_integrals[key]))[mask]
    positions[key] = positions[key][mask]
    # convert to tensors
    positions[key] = torch.Tensor(positions[key])
    transfer_integrals[key] = torch.Tensor(transfer_integrals[key])
    # add data point to dataset list 
    for pos, target in zip(positions[key], transfer_integrals[key]):
        dataset[key].append(Data(pos=pos, y=[key, target]))

source_domain_labels = [0]

# TODO: add shuffle to dataset dicts
# atoms = torch.LongTensor([atomic_numbers[s] for s in atoms])
sep_dataset = [dataset[key][:item] for key, item in len_dataset_domain.items()]
dataset = []
for sublist in sep_dataset:
    dataset.extend(sublist)
shuffle(dataset)# TODO: add a generator

train_set = dataset[:train_len]
val_set = dataset[train_len:train_len+val_len]
test_set = dataset[train_len+val_len:]
train_loader = GeoLoader(train_set, shuffle=True, batch_size=BATCHSIZE)
val_loader = GeoLoader(val_set, shuffle=True, batch_size=VAL_BATCHSIZE)
test_loader = GeoLoader(test_set, shuffle=True, batch_size=VAL_BATCHSIZE)

len_source_domain_train_set = np.sum(np.isin(np.asarray([data.y[0] for data in train_set]), train_labels))
len_source_domain_val_set = np.sum(np.isin(np.asarray([data.y[0] for data in val_set]), train_labels))

########################################################################
if debug==False:
    model = DimeNet_custom(num_classes=1,
                    hidden_channels=128,
                    out_channels=1,
                    num_blocks=6,
                    num_bilinear=8,
                    num_spherical=7,
                    num_radial=6)
# debug
else: 
    model = DimeNet_custom(
                    hidden_channels=2,
                    out_channels=1,
                    num_blocks=2,
                    num_bilinear=8,
                    num_spherical=3,
                    num_radial=1,
                    num_classes=2,
                    alpha=1.0,
                    )
    
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
criterion_1 = F.mse_loss
criterion_2 = F.cross_entropy
weights_train = torch.tensor([1., len_source_domain_train_set/(train_len-len_source_domain_train_set)], dtype=torch.float).to(DEVICE) # weights of cross entropy loss function for each label
weights_val = torch.tensor([1., len_source_domain_val_set/(val_len-len_source_domain_val_set)], dtype=torch.float).to(DEVICE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=10,
                                                        min_lr=0.0000001)

loss_history = []
early_stopper = EarlyStopper(patience=30, min_delta=1)
for epoch in range(1, EPOCHS+1):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss_1, loss_2 = train_grl( epoch, 
                                DEVICE, 
                                model, 
                                criterion_1, 
                                criterion_2, 
                                weights_train,
                                optimizer, 
                                train_loader,
                                source_domain_labels,
                                len_source_domain_train_set)
    
    val_loss_1_lab, val_error_1_lab, val_error_1_unl, val_loss_2 = test_grl( DEVICE, 
                                                                        model, 
                                                                        criterion_1, 
                                                                        criterion_2, 
                                                                        weights_val,
                                                                        val_loader, #TODO: add source domain labels list and len_source domain test set
                                                                        source_domain_labels,
                                                                        len_source_domain_val_set)
    scheduler.step(val_error_1_lab)

    if early_stopper.early_stop(val_loss_1_lab):  
        break

    print(  f'Epoch: {epoch:03d},' 
            f'LR: {lr:.3e},'
            f'train_loss: {loss_1:.7f},'
            f'MAE_lab: {val_error_1_lab:.3f},'
            f'MAE_unl: {val_error_1_unl:.3f}')
    loss_history.append({"train":loss_1, "MAE_lab":val_error_1_lab}) 

########################################################################

# save model 
model_path = os.getcwd() + f'/dimenet/weights/model_{epoch}.pth'
# torch.save(model, model_path)           
with open(model_path, 'wb') as f:
    dill.dump(model, f)

# save test_set
for i in range(test_len):
    torch.save(test_set[i], f'data_{i}.pt')