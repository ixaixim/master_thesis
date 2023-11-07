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


##############
# Dataset and dataloader

debug=True
experiment_num = 9
train_dataset_name = 'debug_penta_100_tetra_100_train'
test_dataset_name = 'debug_penta_25_tetra_25_test'
domain_dict = {'pentacene':0, 'tetracene':1} 
exclude_cols = 0 # to use when test dataset contains atoms that train dataset does not. exclude_cols = number of extra atoms in test_set



num_domains = len(domain_dict)
log_target = True
ordered_domain_dict = OrderedDict(sorted(domain_dict.items(), key=lambda t: t[1]))
# target_domain_k_v = list(ordered_domain_dict.items())[-1] # list of [target_d_name, target_d_int]
# target_domain = target_domain_k_v[1]

now = datetime.now()
formatted_date_time = now.strftime("%b%d_%H-%M-%S")
writer = SummaryWriter(log_dir=f'results/runs/vanilla/exp_{experiment_num}_{formatted_date_time}')

# main Hyperparams
EPOCHS = 250
BATCHSIZE = 512 # NOTE: should be large enough to not be subject to noise, but small enough to guarantee fast enough updates.
VAL_BATCHSIZE = 64
learning_rate = 0.1


# debug 
if debug is True:
    EPOCHS = 100
    BATCHSIZE = 40 # large batchsize seems to be better 
    VAL_BATCHSIZE = 4

print(f'Train dataset: {train_dataset_name}')
print(f'Test dataset: {test_dataset_name}')
# used for training and validation
# for statistics


(mean, std) = get_mean_std_dev(train_dataset_name, log_target=log_target)
target_transforms = Compose([NormalizeTransform(mean, std), MaskExtraFeatures(exclude_cols)])
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

class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift
    
# consider using GCN layers instead of NNConv
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



#%%
import torch.nn.functional as F
node_feat_num = train_set[0].x.shape[-1] # node feature number 
edge_feat_num = train_set[0].edge_attrs.shape[-1] # edge feature number

model = Basemodel(node_feat_num, edge_feat_num)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.5, patience=10,
                                                       min_lr=0.0000001)
early_stopper = EarlyStopper(patience=40, min_delta=0.001)

####################################################
# Training routine
####################################################

def train(epoch):
    # train model for an epoch
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), torch.tensor(data.y, dtype=torch.float).to(device)[:,1].view(-1,1)) # TODO: consider having data.y already a tensor
        loss.backward()
        loss_all += loss.item() * data.num_graphs  
        optimizer.step()
    return loss_all / len(train_loader.dataset) # return average MSE losses over batches

def test(loader):
    # test model on test or validation dataset
    model.eval()
    error = 0
    loss_all = 0
    for data in loader:
        data = data.to(device)
        prediction = model(data)
        loss_all += F.mse_loss(prediction, torch.tensor(data.y, dtype=torch.float).to(device)[:,1].view(-1,1)).item() * data.num_graphs 
        test_y = prediction.cpu().detach().numpy()
        true_y = np.asarray(data.y)[:,1].reshape(-1,1)
        error += np.sum(np.abs(test_y - true_y))
    return loss_all / len(loader.dataset), error / len(loader.dataset) # return average mse and mae loss of dataset


loss_history = []
best = {}
base_path = os.path.join(os.getcwd(), 'NN_conv_net', 'model_weights', 'vanilla')
for epoch in range(1, EPOCHS+1):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train(epoch)
    val_loss, val_error = test(val_loader)

    # early stop
    if early_stopper.early_stop(val_error):  
        print("Process stopped due to early stopping")
        break     
    # checkpointing best model
    if early_stopper.counter == 0: 
        if early_stopper.best_epoch is not None: 
            model_path = os.path.join(base_path, f'exp_{experiment_num}_model_{early_stopper.best_epoch}.pth')
            os.remove(model_path)
        model_path = os.path.join(base_path, f'exp_{experiment_num}_model_{epoch}.pth')
        early_stopper.best_epoch = epoch
        torch.save(model, model_path)
# 
    if epoch==1:
        best['MAE'] = val_error
        best['model_number'] = epoch
        best['model_path'] = model_path
    if val_error < best['MAE']:
        best['MAE'] = val_error
        best['model_number'] = epoch
        best['model_path'] = model_path
    print(f'Epoch: {epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '
          f'Val Loss: {val_loss:.7f}, Val MAE: {val_error:.7f}')
    loss_history.append({"train": loss, "val":val_loss})
    scheduler.step(val_error)

    # logging 
    writer.add_scalar("Loss/train_predictor", loss, epoch)
    writer.add_scalar("Loss/val_source_domain", val_loss, epoch)
    writer.add_scalar("Learning_rate", lr, epoch)
    writer.add_scalar("MAE/val_source_domain", val_error, epoch) # val loss is always in source domain (in vanilla)
print(f"Training finished. Best model at epoch {best['model_number']}")

#%%
def predict(loader):
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

    return np.concatenate(truth), np.concatenate(preds), np.concatenate(domains)

#%%


####################################################
# Check results
####################################################


from sklearn.metrics import mean_absolute_error

print(f'loading best model with weights at path: {best["model_path"]}')
model_path = os.path.join(base_path, f'exp_{experiment_num}_model_{early_stopper.best_epoch}.pth')
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

train_set_mean = np.mean(np.asarray([g.y for g in train_set])[:,0]) 
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
fig.savefig(f"results/figures/vanilla/exp_{experiment_num}_{formatted_date_time}.png")
# %%





     


    

    







    





