# Main of Domain adaptation applied to GNN
# In this main we train on pentacene and unlabeled tetracene 
# Test on: tetracene

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from math import floor
import os
from datetime import datetime
from collections import OrderedDict


from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as GeoLoader
from dataset import MoleculeDataset, NormalizeTransform
from DA_gnn_model import *

from torch.utils.tensorboard import SummaryWriter
from helpers import EarlyStopper, get_mean_std_dev
# reproducibility
torch.manual_seed(41)

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device used: {device}')


##############
# Dataset and dataloader

debug=False
experiment_num = 8
in_domain = False # whether doing in domain or out of domain experiment
train_dataset_name = 'penta_7p5k_tetra_7p5k_DNTT_5k_train'
test_dataset_name = 'DNTT_5k_test'
domain_dict = {'pentacene': 0, 'tetracene': 1, 'DNTT': 2} # keep target domain as last element
source_domains = [domain_dict['pentacene'], domain_dict['tetracene']] # list of source domains, must always be a list of ints



num_domains = len(domain_dict)
log_target = True
ordered_domain_dict = OrderedDict(sorted(domain_dict.items(), key=lambda t: t[1]))
target_domain_k = list(ordered_domain_dict.keys())[-1] # name of target domain
if in_domain is True:
    target_domain_k = None

now = datetime.now()
formatted_date_time = now.strftime("%b%d_%H-%M-%S")
writer = SummaryWriter(log_dir=f'results/runs/DA/exp_{experiment_num}_{formatted_date_time}')

# main Hyperparams
EPOCHS = 250
BATCHSIZE = 512 
VAL_BATCHSIZE = 64
lr = 0.1
# NOTE: large BATCHSIZE is good for accurrate loss, since predictor loss is only applied to labelled elements 
#       (thus reducing the actual batch size)

# debug 
if debug is True:
    EPOCHS = 100
    BATCHSIZE = 10
    VAL_BATCHSIZE = 4

print(f'Train dataset: {train_dataset_name}')
print(f'Test dataset: {test_dataset_name}')
# used for training and validation
# for statistics


(mean, std) = get_mean_std_dev(train_dataset_name, log_target=log_target, target_domain=target_domain_k)
train_dataset = MoleculeDataset(
                                root='data', 
                                filename=train_dataset_name+'.h5', 
                                processed_files_dir='processed/'+train_dataset_name,
                                log_target=log_target, 
                                domains=domain_dict,
                                target_domain = target_domain_k,
                                normalize={'mode':'train'},
                                transform=NormalizeTransform(mean, std)
                                )

# used only for testing 
test_dataset = MoleculeDataset  ( 
                                root='data', 
                                filename=test_dataset_name+'.h5', 
                                processed_files_dir='processed/'+test_dataset_name,
                                log_target=log_target, 
                                domains=domain_dict,
                                normalize={'mode':'test', 'mean':mean, 'std':std},
                                transform=NormalizeTransform(mean, std),
                                )

print(f'Fetching dataset {train_dataset.filename}')
print(f'Fetching dataset {test_dataset.filename}')

# train_set, val_set, test_set = torch.utils.data.random_split(dataset, [0.6,0.2,0.2]) # need higher version of pytorch
val_len = floor(len(train_dataset)*0.25)
train_len = len(train_dataset) - (val_len)
train_set, val_set = torch.utils.data.random_split(train_dataset, [train_len, val_len])
test_set = test_dataset

train_loader = GeoLoader(train_set, batch_size=BATCHSIZE) # can add shuffle
val_loader = GeoLoader(val_set, batch_size=VAL_BATCHSIZE)
test_loader = GeoLoader(test_set, batch_size=VAL_BATCHSIZE)

# count the number of labeled samples in train and val set
source_domain_train_len = np.sum(np.isin(np.asarray([data.y[0] for data in train_set]), source_domains))
source_domain_val_len = np.sum(np.isin(np.asarray([data.y[0] for data in val_set]), source_domains))

# sizes of domains in train_set (use first domain as reference)
sizes_train_set = []
sizes_val_set = []
for i, key in enumerate(ordered_domain_dict.keys()):
    sizes_train_set.append(np.sum(np.asarray([data.y[0] for data in train_set]) == domain_dict[key]))
    sizes_val_set.append(np.sum(np.asarray([data.y[0] for data in val_set]) == domain_dict[key]))
print('\nTraining set description:\n')
for i, (k,v) in enumerate(ordered_domain_dict.items()):
    print(f'Element: {k}')
    print(f'Number of datapoints: {sizes_train_set[i]+ sizes_val_set[i]}')
    print(f'Of which in training set: {sizes_train_set[i]}')
    print(f'Of which in validation set: {sizes_val_set[i]}')

####################
# Training routine
####################

import torch.nn.functional as F
node_feat_num = train_set[0].x.shape[-1] # node feature number 
edge_feat_num = train_set[0].edge_attrs.shape[-1] # edge feature number
model = DA_GNN(node_feat_num, edge_feat_num, num_classes=num_domains)
model = model.to(device)
criterion = F.mse_loss
weights = [sizes_train_set[0]/size for size in sizes_train_set] # weights for loss function
weights = torch.tensor(weights, dtype=torch.float32).to(device) # TODO: adapt to domains
domain_loss = F.cross_entropy 
start = 25
end = 55



# in the following, we use the numbers 1 and 2 to denote the two adversarial networks
# 1: (feature_extractor + label_predictor) network
# 2: (feature_extractore + gradient_reversal_layer + domain_predictor) network  

optimizer_1 = torch.optim.Adam(model.parameters(), lr=lr) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, mode='min',
                                                       factor=0.5, patience=10,
                                                       min_lr=0.0000001)
early_stopper = EarlyStopper(patience=40, min_delta=0.001)

# consider adding leraning rate schedule

def train(epoch):
    model.train()

    running_loss_2 = 0
    running_loss_1 = 0

    # loss1: loss label predictor 
    # loss2: loss domain classifier
    if epoch < start :
        alpha = -1 # i.e. no gradient reversal. Loss = loss1 + loss2
    elif epoch < end: 
        alpha = -1 + 2*(epoch-start) /(end-start) + 0.001  
    else :
        alpha = 1 # i.e. full gradient reversal. Loss = loss1 - loss2
    
    # note: in practice, these constants are also adapted by the Adam algorithm (i.e. we are doing pure SGD DA)
  
    for data in train_loader:
        data = data.to(device)
        output_1, output_2 = model(data, alpha) 
        # criterion loss only on labeled dataset 
        domains = np.asarray(data.y)[:,0]
        mask = np.isin(domains, source_domains)
        num_labeled_graphs = np.sum(mask)
        loss1 = criterion(output_1[mask], torch.FloatTensor(data.y).to(device)[mask,1].view(-1,1)) # loss1 only for labelled instances
        loss2 = domain_loss(output_2, torch.LongTensor((np.asarray(data.y)[:,0]).astype(int)).to(device), weight=weights) # loss2 for all instances

        optimizer_1.zero_grad()
        loss1.backward(retain_graph=True) 
        loss2.backward(retain_graph=False)
        optimizer_1.step()


        running_loss_1 += loss1.item() * num_labeled_graphs
        running_loss_2 += loss2.item() * data.num_graphs
        
    loss_1 = running_loss_1 / source_domain_train_len
    loss_2 = running_loss_2 / len(train_loader.dataset)
    # return the average loss per epoch
    return loss_1, loss_2

def test(loader):
    # return average mse, mae loss of labeled data.
    # return binary cross entropy loss of unlabeled data.
    # return also validation loss on unlabeled data
    model.eval()
    error_source_dom = 0
    error_target_dom = 0
    loss_source_dom = 0
    ce_loss_all = 0
    for data in loader:
        data = data.to(device)
        prediction, domain_class = model(data)
        domains = np.asarray(data.y)[:,0]
        mask = np.isin(domains, source_domains)        
        num_labeled_graphs = np.sum(mask)

        # mse loss
        loss_source_dom += criterion(prediction[mask], torch.tensor(data.y, dtype=torch.float).to(device)[mask,1].view(-1,1)).item() * num_labeled_graphs

        # ce loss
        ce_loss_all += domain_loss(domain_class, torch.LongTensor((np.asarray(data.y)[:,0]).astype(int)).to(device), weight=weights).item() * data.num_graphs

        # mae loss 
        test_y = prediction.detach().cpu().numpy()
        true_y = np.asarray(data.y)[:,1].reshape(-1,1)
        error_source_dom += np.sum(np.abs(test_y - true_y)[mask])
        error_target_dom += np.sum(np.abs(test_y - true_y)[~mask]) # TODO: log this quantity during training
    return loss_source_dom / source_domain_val_len, error_source_dom / source_domain_val_len, error_target_dom / (val_len-source_domain_val_len), ce_loss_all / val_len


# run training
models = []
best = {}
base_path = os.path.join(os.getcwd(), 'NN_conv_net', 'model_weights', 'DA')
for epoch in range(1, EPOCHS+1):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss_1, loss_2 = train(epoch)
    val_loss, val_error_source_dom, val_error_target_dom, val_bin_ce_loss = test(val_loader)
        # early stop (only after full DA is working)
    if (epoch >= end) | (epoch==1):
        if early_stopper.early_stop(val_error_source_dom):  
            print("Process stopped due to early stopping")
            break     
        # checkpointing best model
        if early_stopper.counter == 0: 
            if early_stopper.best_epoch is not None: # remove the previous best model from memory
                model_path = os.path.join(base_path, f'exp_{experiment_num}_model_{early_stopper.best_epoch}.pth')
                os.remove(model_path)
            model_path = os.path.join(base_path, f'exp_{experiment_num}_model_{epoch}.pth')
            early_stopper.best_epoch = epoch
            torch.save(model, model_path)

    if epoch==1: # initialize best epoch
        best['MAE'] = val_error_source_dom
        best['model_number'] = epoch
        best['model_path'] = model_path
    if (epoch >= end):
        if val_error_source_dom < best['MAE']: 
            best['MAE'] = val_error_source_dom
            best['model_number'] = epoch
            best['model_path'] = model_path
    scheduler.step(val_error_source_dom)



    print(f'Epoch: {epoch:03d}, LR: {lr:.7f}, '
            f'Predictor loss: {loss_1:.7f}, Domain Loss: {loss_2:.7f}, '
            f'val_MAE_source_dom: {val_error_source_dom:.7f}, val_MAE_target_dom: {val_error_target_dom:.7f}')
    writer.add_scalar("Loss/train_predictor", loss_1, epoch)
    writer.add_scalar("Loss/train_domain", loss_2, epoch)
    writer.add_scalar("Loss/val_source_domain", val_loss, epoch)
    writer.add_scalar("MAE/val_source_domain", val_error_source_dom, epoch)
    writer.add_scalar("MAE/val_target_domain", val_error_target_dom, epoch)
    writer.add_scalar("Learning_rate", lr, epoch)

print(f"Training finished. Best model at epoch {best['model_number']}")


def predict(loader):
    model.eval()
    preds = []
    truth = []
    domains = []
    for data in loader:
        data = data.to(device)
        preds.append(((model(data))[0].cpu().detach().numpy()).reshape(-1,1))
        truth.append(np.asarray(data.y)[:,1].reshape(-1,1))
        domains.append(np.asarray(data.y)[:,0].reshape(-1,1))

    return np.concatenate(truth), np.concatenate(preds), np.concatenate(domains)

####################################################
# Check results
####################################################
from sklearn.metrics import mean_absolute_error

print(f'loading best model with weights at path: {best["model_path"]}')
model_path = os.path.join(base_path, f'exp_{experiment_num}_model_{early_stopper.best_epoch}.pth' )
model = torch.load(model_path)


y_true, y_pred, domains = predict(test_loader)
true_domains = np.asarray([g.y for g in test_set])[:,0]

test_set_mae = {}
y_true_dom = {}
y_pred_dom = {}
y_true = y_true*std + mean
y_pred = y_pred*std + mean
for k,v in domain_dict.items():
    y_true_dom[k], y_pred_dom[k] = y_true[v==true_domains], y_pred[v==true_domains]
    if len(y_true_dom[k]) > 0 and len(y_pred_dom[k]) > 0:
        test_set_mae[k] = mean_absolute_error(y_true_dom[k], y_pred_dom[k])

mask = np.isin(np.asarray([g.y for g in train_set])[:,0], source_domains) # NOTE: the mask is over the source domains, to give an average dumb mean
source_domain_targets = np.asarray([g.y for g in train_set])[mask,1]
train_set_mean = np.mean(source_domain_targets) 
dumb_mae = mean_absolute_error(y_true, np.ones(len(y_true))*train_set_mean)

logging_dict = {}
logging_dict['metrics/baseline_MAE'] = dumb_mae
for k,v in test_set_mae.items():
    new_k = 'metrics/' + k + '_MAE'
    logging_dict[new_k] = v
writer.add_hparams({'bsize': BATCHSIZE, 'epochs': epoch}, logging_dict)

writer.flush()
writer.close()



fig, axis = plt.subplots()
if log_target:
    axis.set_title(f"Mean-baseline MAE {dumb_mae:.5f} ")
    axis.set_xlabel(r"log$V_{i,j}$")
    axis.set_ylabel(r"log$\hat{V}_{i,j}$")
    xmin, xmax = -20, 0
    ymin, ymax = -20, 0

axis.grid()
axis.set_xlim(xmin,xmax)
axis.set_ylim(ymin,ymax)
axis.set_aspect(1)
colors = ["red", "green", "blue", "yellow"]

for i,k in enumerate(test_set_mae.keys()):
    axis.scatter(y_true_dom[k], y_pred_dom[k], 
                 marker='o', facecolors='none', edgecolors=colors[i], alpha=0.4,
                 label=k+f' MAE: {test_set_mae[k]:.3f}')
axis.legend(loc='upper left')
axis.plot((xmin, xmax), (ymin, ymax), color='black')


# plt.show()
fig.savefig(f"results/figures/DA/exp_{experiment_num}_{formatted_date_time}.png")
# %%





     


    

    







    





