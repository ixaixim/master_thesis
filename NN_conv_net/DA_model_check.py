#%% 
import os
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
from torch import Tensor
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.transforms import Compose
from dataset import MoleculeDataset, NormalizeTransform, MaskExtraFeatures
from DA_gnn_model import *

from torch.utils.tensorboard import SummaryWriter
from helpers import EarlyStopper, get_mean_std_dev
# reproducibility
torch.manual_seed(41)

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device used: {device}')


# from experiment: 
experiment_num = 2
train_dataset_name = 'penta_10k_tetra_10k_train'
test_dataset_name = 'penta_2p5k_tetra_2p5k_test'
domain_dict = {'pentacene':0, 'tetracene':1} 
exclude_cols = 0 # to use when test dataset contains atoms that train dataset does not. exclude_cols = number of extra atoms in test_set
# set manually
model_to_load = 104
mean, std = -5.441199779510498, 3.584162950515747


num_domains = len(domain_dict)
log_target = True
ordered_domain_dict = OrderedDict(sorted(domain_dict.items(), key=lambda t: t[1]))

# (mean, std) = get_mean_std_dev(train_dataset_name, log_target=log_target)
target_transforms = Compose([NormalizeTransform(mean, std), MaskExtraFeatures(exclude_cols)])
# train set used for mean computation
train_dataset = MoleculeDataset(
                                root='data', 
                                filename=train_dataset_name+'.h5', 
                                processed_files_dir='processed/'+train_dataset_name,
                                log_target=log_target, 
                                domains=domain_dict,
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
                                transform=target_transforms,
                                )
VAL_BATCHSIZE = 1
BATCHSIZE = 40 # large batchsize seems to be better 

val_len = floor(len(train_dataset)*0.25)
train_len = len(train_dataset) - (val_len)
train_set, val_set = torch.utils.data.random_split(train_dataset, [train_len, val_len])
test_set = test_dataset

test_loader = GeoLoader(test_set, batch_size=VAL_BATCHSIZE)



def predict(loader, mask):
    # if using a transformation on target, make sure to transform back
    model.eval()
    preds = []
    truth = []
    domains = []
    for data in loader:
        data = data.to(device)
        preds.append((model(data).cpu().detach().numpy()).reshape(-1))
        truth.append(np.asarray(data.y)[:,1].reshape(-1,1))
        domains.append(np.asarray(data.y)[:,0].reshape(-1,1))
        if len(preds) == mask:
            break

    return np.concatenate(truth), np.concatenate(preds), np.concatenate(domains)


class Basemodel(nn.Module):
    def __init__(self,
    nodefeat_num=3, edgefeat_num=1,
    nodeembed_to=4, edgeembed_to=4):
        super().__init__()
        ## Embeddings
        self._node_embedding = nn.Linear(nodefeat_num, nodeembed_to)
        self._node_embednorm = (nn.BatchNorm1d(nodeembed_to))
        self._edge_embedding = nn.Linear(edgefeat_num, edgeembed_to)
        self._edge_embednorm = (nn.BatchNorm1d(edgeembed_to))

        self.NNs = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Graph Convolutions
        self.NNs.append(nn.Sequential(nn.Linear(edgeembed_to, nodeembed_to**2)))
        self.convs.append(NNConv(nodeembed_to, nodeembed_to, self.NNs[-1]))
        self.bns.append(BatchNorm(nodeembed_to))

        self.NNs.append(nn.Sequential(nn.Linear(edgeembed_to, nodeembed_to**2)))
        self.convs.append(NNConv(nodeembed_to, nodeembed_to, self.NNs[-1]))
        self.bns.append(BatchNorm(nodeembed_to))

        self.NNs.append(nn.Sequential(nn.Linear(edgeembed_to, nodeembed_to**2)))
        self.convs.append(NNConv(nodeembed_to, nodeembed_to, self.NNs[-1]))
        self.bns.append(BatchNorm(nodeembed_to))

        self.NNs.append(nn.Sequential(nn.Linear(edgeembed_to, nodeembed_to**2)))
        self.convs.append(NNConv(nodeembed_to, nodeembed_to, self.NNs[-1]))
        self.bns.append(BatchNorm(nodeembed_to))

        self.NNs.append(nn.Sequential(nn.Linear(edgeembed_to, nodeembed_to**2)))
        self.convs.append(NNConv(nodeembed_to, nodeembed_to, self.NNs[-1]))
        self.bns.append(BatchNorm(nodeembed_to))


        self.act = nn.ReLU() # ShiftedSoftplus() : not that good 
        # first, pass the initial size of the nodes
         # and their output-size
         # the transformation NN h_\Theta
        ## Pooling and actuall prediction NN
        self._pooling = [global_mean_pool, global_max_pool] # takes batch.x and batch.batch as args
        # shape of one pooling output: [B,F], where B is batch size and F the number of node features.
        # shape of concatenated pooling outputs: [B, len(pooling)*F]
        self._predictor = nn.Sequential(
            nn.Linear(nodeembed_to*len(self._pooling), nodeembed_to*4),
            nn.ReLU(),
            nn.BatchNorm1d(nodeembed_to*4),
            nn.Linear(nodeembed_to*4, nodeembed_to*3),
            nn.ReLU(),
            nn.BatchNorm1d(nodeembed_to*3),
            nn.Linear(nodeembed_to*3, nodeembed_to*2),
            nn.ReLU(),
            nn.BatchNorm1d(nodeembed_to*2),
            nn.Linear(nodeembed_to*2, 1)
        )        
    def forward(self, batch: Batch):
        # NOTE: a batch is just a large disconnected graph, where each sub-graph is a batch element
        node_features, edges, edge_features, batch_vector = \
            batch.x.float(), batch.edge_index, batch.edge_attrs.float(), batch.batch
        ## feature embedding
        node_features = self._node_embednorm(
            self._node_embedding(node_features))
        edge_features = self._edge_embednorm(
            self._edge_embedding(edge_features))
        

        ## graph convolutions
        for step in range(len(self.convs)):
            node_features = self.bns[step](self.act(self.convs[step](node_features, edges, edge_features)))

        ## pooling
        # if pooled_graph_nodes
        pooled_graph_nodes = torch.cat([p(node_features, batch_vector) for p in self._pooling], axis=1) 
        outputs = self._predictor(pooled_graph_nodes)
        return outputs         



# end of copy
################################################################################
#%%
base_path = os.path.join(os.getcwd(), 'NN_conv_net', 'model_weights', 'vanilla')

model_path = os.path.join(base_path, f'exp_{experiment_num}_model_{model_to_load}.pth')

model = torch.load(model_path, map_location='cpu')


from sklearn.metrics import mean_absolute_error

mask = 2500
y_true, y_pred, true_domains = predict(test_loader, mask)
true_domains = np.asarray([g.y for g in test_set[:mask]])[:,0]

test_set_mae = {}
y_true_dom = {}
y_pred_dom = {}
y_true = y_true*std + mean
y_pred = y_pred*std + mean
for k,v in domain_dict.items():
    y_true_dom[k], y_pred_dom[k] = y_true[v==true_domains], y_pred[v==true_domains]
    if len(y_true_dom[k]) > 0 and len(y_pred_dom[k]) > 0:
        test_set_mae[k] = mean_absolute_error(y_true_dom[k], y_pred_dom[k])

# train_set_mean = np.mean(np.asarray([g.y for g in train_set])[:,0]) 
# dumb_mae = mean_absolute_error(y_true, np.ones(len(y_true))*train_set_mean)

fig, axis = plt.subplots()
if log_target:
    # axis.set_title(f"Mean-baseline MAE {dumb_mae:.5f} ")
    axis.set_xlabel(r"log$V_{i,j}$")
    axis.set_ylabel(r"log$\hat{V}_{i,j}$")
    xmin, xmax = -20, 0
    ymin, ymax = -20, 0

axis.grid()
axis.set_xlim(xmin,xmax)
axis.set_ylim(ymin,ymax)
axis.set_aspect(1)
colors = ["red", "green", "blue", "yellow"]

# for i,k in enumerate(test_set_mae.keys()):
#     axis.scatter(y_true_dom[k], y_pred_dom[k], 
#                  marker='o', facecolors='none', edgecolors=colors[i], alpha=0.4,
#                  label=k+f' MAE: {test_set_mae[k]:.3f}')
k = 'pentacene'
i=0
axis.scatter(y_true_dom[k], y_pred_dom[k], 
                marker='o', facecolors='none', edgecolors=colors[i], alpha=0.4,
                label=k+f' MAE: {test_set_mae[k]:.3f}')

axis.legend(loc='upper left')
axis.plot((xmin, xmax), (ymin, ymax), color='black')

fig.savefig(f"exp_{experiment_num}.png")
