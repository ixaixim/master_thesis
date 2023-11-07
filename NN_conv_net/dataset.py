import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import h5py
from torch_geometric.data import Dataset, Data
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import dataset
import random
random.seed(42)

# MBTR
from ase import Atoms
from dscribe.descriptors import MBTR
import ase

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


'''
Implement Dataset class for managing datasets, transforming and loading instances.
Decouple the data pipeline process from the main file.
'''


# TODO: remove molecules whose distance between center of mass is greater than 18 angstrom. (see data_analysis.ipynb)


# NOTE: must implement __init__ and __call__ functions for Transform
# NOTE: consider using InMemoryDataset, since data should easily be stored on CPU
#       in that case there are a couple of extra classes that need to be implememented.

class MoleculeDataset(Dataset):
    '''
    Dataset class for managing and transforming molecule datasets. 
    Decouple the data pipeline process from the main file.
    '''
    def __init__(
                self, 
                root, 
                filename=['john_pentacene.h5'], 
                processed_files_dir=None, 
                transform=None,
                pre_transform=None, 
                log=False,
                log_target=True, 
                normalize=None, 
                domains=None,
                target_domain=None,
                ):
        self.root = root
        self.filename = filename
        self.processed_files_dir = processed_files_dir # name of directory where processed data is stored
        self.groups = []
        self.log_target = log_target # log transform of target, discard inf values
        self.domains = domains # add domain label dictionary for use in Domain Adaptation
        self.target_domain = target_domain # target domain for masking when doing transformations on training targets
        self.normalize = normalize # normalize the target. Is a dictionary with keys: 'mode' (i.e. train/test), 'mean', 'std'

        super(MoleculeDataset, self).__init__(root, transform, pre_transform, log)
    
    # TODO: maybe I have to use @setter 
    @property 
    def processed_dir(self) -> str: 
        if self.processed_files_dir:
            return os.path.join(self.root, self.processed_files_dir)
        else: 
            super().processed_dir

    @property
    def raw_file_names(self):
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading."""
        return self.filename

    @property
    def processed_file_names(self):
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing."""
        # print(f"Extracting processed files from: {self.processed_dir}")
        data = [f for f in os.listdir(self.processed_dir) if f.startswith('data_')]
        return data
        
    def process(self):
        # Read the files content as numpy array.

        # TODO: DO FOR EACH GROUP IN H5 FILE AND create dictionary of positions and transfer integrals, 
        #       ADDING ALSO A DOMAIN array BASED ON A DICTIONARY {'group_name':label_num} (e.g. {'pentacene':0})
        #       FOR STATISTICS: DO NOT READ TRANSFER INTEGRALS OF TARGET DOMAIN 
        #                       (CAN ACHIEVE THIS BY ADDING AN ATTRIBUTE DICTIONARY WITH {'group_name':label_num} OF TARGET DOMAIN)
        #                       AND AN IF STATEMENT IN THE STATISTIC COMPUTATION PART.
        # read pentacene file (or multiple files if necessary)
        atoms = {}
        positions = {}
        transfer_integrals = {}
        with h5py.File(self.raw_paths[0], "r") as file: 
            self.groups = list(file.keys())
            for group in self.groups:
                # list top-level groups/datasets in file
                print(file.keys())

                # get content and convert to numpy arrays
                atoms[group] = file[group].get("atoms")[()].astype(np.str)
                positions[group] = file[group].get("positions")[()]
                transfer_integrals[group] = file[group].get("transfer_integrals")[()] 

        # Log of target
        print(f'Transforming target: log_target set to {self.log_target}')
        if self.log_target: # discard instances with inf values of log (i.e. discard transfer_integral==0)
            for key in transfer_integrals:
                mask = ~(transfer_integrals[key]==0.)
                transfer_integrals[key] = np.log10(np.abs(transfer_integrals[key]))[mask] #  TODO: log 10
                positions[key] = positions[key][mask]
        else:
            for key in transfer_integrals:
                transfer_integrals[key] = np.abs(transfer_integrals[key])

 
        ############################################################################
        # creation of Data object

        atom_to_num = {'H': 1, 'C': 6, 'S':16} # atom to atomic number
        all_atoms = np.concatenate(list(atoms.values())) # needed for one-hot encoding of atom numbers
        all_atomic_nums = np.asarray([atom_to_num[atom] for atom in all_atoms])[:, np.newaxis] # keep as numpy for later use
        unique_values = np.unique(all_atomic_nums) 
        num_unique = len(unique_values)

        # In the loop we extract the nodes' embeddings, edges connectivity 
        #   and label for a graph, process the information and put it in a Data
        #   object, then we add the object to a list
        data_list = []
        for key in transfer_integrals:
            # for index, (pos, target) in tqdm(enumerate(zip(positions, transfer_integrals)), total=len(positions)):
            for index, (pos, target) in enumerate(zip(positions[key], transfer_integrals[key])):
                
                ############################################################
                # Node features
                num_nodes = len(atoms[key])

                    # atomic number and electronegativity
                atom_to_num = {'H': 1, 'C': 6, 'S':16} # atom to atomic number
                atom_to_en = {'H': 2.2, 'C': 2.55, 'S':2.58} # atom to electronegativity
                atomic_nums = np.asarray([atom_to_num[atom] for atom in atoms[key]])[:, np.newaxis] # keep as numpy for later use
                atomic_nums_torch = torch.tensor(atomic_nums, dtype=torch.double)
                electroneg = torch.tensor(np.asarray([atom_to_en[atom] for atom in atoms[key]])[:, np.newaxis], dtype=torch.double)
                    
                    # one-hot array for atomic element
                mapping = {val: i for i, val in enumerate(unique_values)}
                atomic_nums_ordered = np.array([mapping[val] for val in atomic_nums.squeeze()])
                one_hot_atomic_nums = np.eye(num_unique)[atomic_nums_ordered]
                one_hot_atomic_nums = torch.tensor(one_hot_atomic_nums, dtype=torch.double)

                    # one-hot encoding of atoms[key] belonging to molecule 1 or 2
                belongs_to_mol = np.array([1 if i < len(atoms[key])//2 else 0 for i in range(len(atoms[key]))])
                belongs_to_mol = np.eye(2)[belongs_to_mol]
                belongs_to_mol = torch.tensor(belongs_to_mol, dtype=torch.double)

 
                # available: atomic_nums_torch, electroneg, one_hot_atomic_nums, feat_mbtr, belongs_to_mol
                # IMPORTANT: keep always one_hot_atomic_nums as last column. It is necessary for transform MaskExtraFeatures
                node_attrs = torch.cat([atomic_nums_torch, electroneg, belongs_to_mol, one_hot_atomic_nums], dim=1)
                
                ############################################################

                # Edge index
                a = np.arange(len(atoms[key]))
                edges = np.array(np.meshgrid(a,a)).T.reshape(-1,2).T
                edges = torch.tensor(edges, dtype=torch.int64)
    
                # masking the interaction part
                interaction_part = False # whether to use only interaction part
                if interaction_part:
                    half_len = len(atoms[key])//2
                    mask = ((edges[0] < half_len) & (edges[1] < half_len)) | ((edges[0] >= half_len) & (edges[1] >= half_len))
                    edges = edges[:, ~mask]
                

                # Edge features
                    # shape [N', D'] N': number of edges, D': number of edge features
                    # cm matrix and bond matrix 
                    # available: distances, cm, is_single, is_aromatic
                distances = pairwise_distances(pos)
                cm = (atomic_nums*atomic_nums.T) / distances
                np.fill_diagonal(cm, 0.5*atomic_nums**2.4)
                cm = torch.tensor(cm.flatten()[:, np.newaxis], dtype=torch.double)
                distances = torch.tensor(distances.flatten()[:, np.newaxis], dtype=torch.double) # TODO: convert already before in tensor to avoid confusion

                # is_single, is_aromatic = pentacene_bond_matrix('NN_conv_net/data/raw/john_pentacene.h5')
                # is_single = torch.tensor(is_single.flatten()[:, np.newaxis], dtype=torch.double)
                # is_aromatic = torch.tensor(is_aromatic.flatten()[:, np.newaxis], dtype=torch.double)

                if interaction_part:
                    edge_attr = torch.cat([distances[~mask], cm[~mask]], dim=1) # TODO: try 1/distances (gives more importance to small distances.)
                else:
                    edge_attr = torch.cat([distances, cm], dim=1)

                # Target: domain label # TODO check
                if isinstance(self.domains[key], int):
                    y = [self.domains[key], target]
                else:
                    y = torch.tensor(target) 

                graph = Data(x=node_attrs,
                        edge_index=edges,
                        edge_attrs=edge_attr, 
                        y=y)
                data_list.append(graph)
        
        random.shuffle(data_list)

        for index, graph in enumerate(data_list):
            torch.save(graph, 
                os.path.join(self.processed_dir,
                                f'data_{index}.pt'))

    def len(self):
        return len(self.processed_file_names)
        
    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        data = torch.load(os.path.join(self.processed_dir,
                            f'data_{idx}.pt'))
        # if self.transform is not None: # transform is called automatically
        #     data = self.transform(data)
        return data
    
class NormalizeTransform: 
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std       
    def __call__(self, data):
        y_normalized = (data.y[1] - self.mean) / self.std
        transformed_data = Data(x=data.x,
                            edge_index=data.edge_index,
                            edge_attrs=data.edge_attrs, 
                            y=[data.y[0], y_normalized]) #BUG: check if you can access and pass **data.attrs
        return transformed_data

class MaskExtraFeatures: 
    # masks the extra columns in one_hot_atomic_nums
    # this is because vanilla network is unaware of number of atoms in test set, 
    # and its training is better with fewer features
    def __init__(self, num_columns):
        self.num_columns = num_columns
    def __call__(self, data):
        if self.num_columns == 0:
            return data.clone()
        else:
            transformed_data = Data(x=data.x[:,:-self.num_columns],
                            edge_index=data.edge_index,
                            edge_attrs=data.edge_attrs, 
                            y=data.y)
        return transformed_data
    
    

# notes: 
# Graph is fully connected -> 72*72 edges (also self-connections)
# edge features are the coulomb matrix and bond matrix (note that are symmetric, therefore can exploit the bidirectionality)

def pentacene_bond_matrix(filename):
    with h5py.File(filename, "r") as file:
        # get content and convert to numpy arrays
        atoms = file.get("atoms")[()].astype(str)
    # construct bond matrix:
    bonds = np.zeros((len(atoms), len(atoms)))
    arom = 1.5
    single = 1 
    counter = 22

    for i in range(len(atoms)):
        # aromatic
        if ((i%4 == 0) and ((i<=16) or ((i>=36) & (i<=52)))):
            bonds[i, i+1],   bonds[i+1, i] = (arom, arom) # (0,1)
            bonds[i, i+2],   bonds[i+2, i] = (arom, arom) # (0,2)
            bonds[i+3, i+1], bonds[i+1, i+3] = (arom, arom) # (3,1)
            bonds[i+3, i+5], bonds[i+5, i+3] = (arom, arom) # (3,5)
            bonds[i+4, i+2], bonds[i+2, i+4] = (arom, arom) # (4,2)
            bonds[i+4, i+5], bonds[i+5, i+4] = (arom, arom) # (4,5)    
        # single
        if ((i<=20) or ((i>=36) & (i<=56))):
            if (i%4==2): 
                bonds[i,i+counter], bonds[i+counter, i] = (single, single)
            if (i%4==3):
                bonds[i,i+counter], bonds[i+counter,i] = (single, single)
                counter-=2 
        if i==22:
            counter=22 
    

        # boundary bonds
        bonds[0,22], bonds[22,0] = (single, single)
        bonds[1,23], bonds[23,1] = (single, single)
        bonds[20,34], bonds[34,20] = (single, single)
        bonds[21,35], bonds[35,21] = (single, single)

        bonds[36,58], bonds[58,36] = (single, single)
        bonds[37,59], bonds[59,37] = (single, single)
        bonds[56, 70], bonds[70, 56] = (single, single)
        bonds[57,71], bonds[71,57] = (single, single)  

        is_aromatic = np.zeros((len(atoms), len(atoms)))
        is_aromatic[np.where(bonds==arom)] = 1
        is_single = np.zeros((len(atoms), len(atoms)))
        is_single[np.where(bonds==single)] = 1

    return is_single, is_aromatic 
                    

