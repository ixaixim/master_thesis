import numpy as np
import h5py
import torch
from torch import LongTensor, Tensor
from ase.data import atomic_numbers


# domains: 
    # pentacene: 0
    # tetracene: 1
    # DNTT: 2
def get_atoms(domain):
    '''
    takes an integer domain as input and returns a PyTorch LongTensor containing the atomic numbers of the atoms in a molecular structure.
    '''
    if domain==0:
        name="john_pentacene"
    elif domain==1:
        name="john_tetracene"
    else:
        print("Domain not available")
    with h5py.File("/home/ge96sur/john_michael_domain_adaptation/data/raw/"+name+".h5", "r") as file:
        atoms = file.get("atoms")[()].astype(str)
    atoms = LongTensor([atomic_numbers[s] for s in atoms])
    return atoms

def train(epoch, device, model, criterion, optimizer, loader):
    model.train()
    running_loss = 0
    domain_to_atoms = {0: get_atoms(0).to(device), 1: get_atoms(1).to(device)} # NOTE: consider passing this dict to function
    atoms = torch.empty((0,), dtype=torch.int8).to(device)
    for data in loader:
        for y in data.y:
            atoms = torch.cat((atoms, domain_to_atoms[y[0]]))
        atoms = atoms.to(device)
        data = data.to(device)
        output = model(z=atoms, pos=data.pos, batch=data.batch) 
        loss = criterion(output, Tensor(data.y)[:,1].to(device).view(-1,1)) 
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        running_loss += loss.item() * data.num_graphs
    loss = running_loss / len(loader.dataset)
    torch.cuda.empty_cache()
    # return the average loss per epoch
    return loss

def test(device, model, criterion, loader):
    # return average mse, mae loss.
    model.eval()
    running_loss = 0  
    running_MAE = 0
    domain_to_atoms = {0: get_atoms(0).to(device), 1: get_atoms(1).to(device)}
    atoms = torch.empty((0,), dtype=torch.int8).to(device)
    for data in loader:
        for y in data.y:
            atoms = torch.cat((atoms, domain_to_atoms[y[0]]))
        data = data.to(device)
        output = model(z=atoms, pos=data.pos, batch=data.batch)
        # mse loss 
        loss = criterion(output, torch.tensor(data.y).to(device)[:,1].view(-1,1))
        running_loss += loss.item() * data.num_graphs
        # mae loss
        test_y = output.detach().cpu().numpy()
        data = data.to('cpu')
        true_y = np.asarray(data.y)[:,1].reshape(-1,1)
        running_MAE += np.sum(np.abs(test_y - true_y))

    loss = running_loss / len(loader.dataset)
    MAE = running_MAE / len(loader.dataset)
    torch.cuda.empty_cache()
    return loss, MAE

def train_grl(epoch, 
              device, 
              model, 
              criterion_1, 
              criterion_2, 
              weights,
              optimizer, 
              loader,
              source_domain_labels,
              len_source_domain_dataset
              ):
    '''
    The train_grl function performs one epoch of training for a graph neural network model using the Gradient Reversal Layer (GRL) approach for domain adaptation.

    Parameters: 
    epoch (int): The current epoch of the training process.
    device (torch.device): The device on which to perform computations.
    model (torch.nn.Module): The neural network model to be trained.
    criterion_1 (torch.nn.Module): The loss function used for the domain-specific task.
    criterion_2 (torch.nn.Module): The loss function used for the domain adaptation task.
    weights (torch.Tensor): The weight vector used to balance the contribution of each class in the domain adaptation loss.
    optimizer (torch.optim.Optimizer): The optimization algorithm used to update the model's parameters.
    loader (torch.utils.data.DataLoader): The data loader for the training set.
    source_domain_labels (List[int]): A list of integers representing the labels of the source domains.
    len_source_domain_dataset (int): The length of the source domain training dataset.
    '''
    model.train()
    running_loss_1 = 0
    running_loss_2 = 0
    domain_to_atoms = {0: get_atoms(0), 1: get_atoms(1)} # TODO: modify
    atoms = torch.empty((0,), dtype=torch.int8)

    start = 25
    end = 100
    if epoch < start:
        alpha = -1 # alpha = -1 i.e. no gradient reversal. Loss = loss1 + loss2
    elif epoch < end: 
        alpha = -1 + 2*(epoch-start) /(end-start) + 0.001  
    else :
        alpha = 1 # alpha = 1 i.e. full gradient reversal. Loss = loss1 - loss2
        
    for data in loader:
        for y in data.y:
            atoms = torch.cat((atoms, domain_to_atoms[y[0]]))
        data = data.to(device)
        domains = np.asarray(data.y)[:,0]
        mask = np.isin(domains, source_domain_labels)

        output_1, output_2 = model(z=atoms, pos=data.pos, alpha=alpha, batch=data.batch)
        output_1, output_2 = (output_1.squeeze(dim=0), output_2.squeeze(dim=0))
        
        truth_1 = Tensor(data.y)[mask,1].view(-1,1).to(device)
        truth_2 = torch.LongTensor((np.asarray(data.y, dtype=int)[:,0])).to(device) # TODO: consider passing the domain label already as integer (in Data object creation), to avoid conversion (for cleaner code)
        loss_1 = criterion_1(output_1[mask], truth_1) 
        loss_2 = criterion_2(output_2, truth_2, weight=weights)
        optimizer.zero_grad()
        loss_1.backward(retain_graph=True) 
        loss_2.backward()
        optimizer.step()
        running_loss_1 += loss_1.item() * np.sum(mask)
        running_loss_2 += loss_2.item() * data.num_graphs
    loss_1 = running_loss_1 / len_source_domain_dataset
    loss_2 = running_loss_2 / len(loader.dataset)
    # return the average loss per epoch
    return loss_1, loss_2

def test_grl( device, 
              model, 
              criterion_1,
              criterion_2,
              weights,
              loader,
              source_domain_labels, 
              len_source_domain_dataset):# NOTE: this is the length of the elements OF test set from the source domain.
    # return average mse, mae loss of labeled data.
    # return binary cross entropy loss of unlabeled data.
    # return also validation loss on unlabeled data
    model.eval()
    running_loss_1_lab = 0
    running_error_1_lab = 0
    running_error_1_unl = 0
    running_loss_2 = 0
    domain_to_atoms = {0: get_atoms(0), 1: get_atoms(1)}
    atoms = torch.empty((0,), dtype=torch.int8)
    for data in loader:
        for y in data.y:
            atoms = torch.cat((atoms, domain_to_atoms[y[0]]))
        data = data.to(device)
        output_1, output_2 = model(z=atoms, pos=data.pos, alpha=None, batch=data.batch)
        output_1, output_2 = (output_1.squeeze(dim=0), output_2.squeeze(dim=0))

        domains = np.asarray(data.y)[:,0]
        mask = np.isin(domains, source_domain_labels)        
        num_labeled_graphs = np.sum(mask)

        truth_1 = Tensor(data.y)[mask,1].view(-1,1).to(device)
        truth_2 = torch.LongTensor((np.asarray(data.y, dtype=int)[:,0])).to(device) # TODO: consider passing the domain label already as integer (in Data object creation), to avoid conversion (for cleaner code)

        # mse loss (labeled)
        loss_1_lab = criterion_1(output_1[mask], truth_1)

        # ce loss
        loss_2 = criterion_2(output_2, truth_2, weight=weights).to(device)

        # mae loss (both labeled and unlabeled)
        test_y = output_1.detach().cpu().numpy()
        true_y = np.asarray(data.y)[:,1].reshape(-1,1)
        train_len = len(loader.dataset) 

        running_loss_1_lab += loss_1_lab.item() * num_labeled_graphs
        running_loss_2 += loss_2.item() * data.num_graphs
        running_error_1_lab += np.sum(np.abs(test_y - true_y)[mask])
        running_error_1_unl += np.sum(np.abs(test_y - true_y)[~mask])  # TODO: log this quantity during training

    loss_1_lab = running_loss_1_lab / len_source_domain_dataset
    loss_2 = running_loss_2 / train_len
    error_1_lab = running_error_1_lab / len_source_domain_dataset
    error_1_unl = running_error_1_unl / (train_len-len_source_domain_dataset)

    return loss_1_lab, error_1_lab, error_1_unl, loss_2

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False