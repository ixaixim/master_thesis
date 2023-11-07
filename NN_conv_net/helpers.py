import numpy as np
import h5py
import os


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.best_epoch = None # this attribute is set in the main

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def get_mean_std_dev(filename, log_target=True, target_domain=None):
    '''
    # Calculates mean and standard deviation of the dataset.
    ## Args:
    1. filename (str) : path of the file containing the dataset.
    2. log_target (bool) : whether to calculate mean and std-deviation of log(targets).
    3. target_domain (float tuple) : min and max values of targets in the dataset.
    '''
    atoms = {}
    positions = {}
    transfer_integrals = {}
    path = os.path.join('data', 'raw', filename+'.h5')
    with h5py.File(path, "r") as file: 
        groups = list(file.keys())
        for group in groups:
            # list top-level groups/datasets in file
            print(file.keys())

            # get content and convert to numpy arrays
            atoms[group] = file[group].get("atoms")[()].astype(np.str)
            positions[group] = file[group].get("positions")[()]
            transfer_integrals[group] = file[group].get("transfer_integrals")[()] 

        # Log of target
        print(f'Transforming target: log_target set to {log_target}')
        if log_target: # discard instances with inf values of log (i.e. discard transfer_integral==0)
            for key in transfer_integrals:
                mask = ~(transfer_integrals[key]==0.)
                transfer_integrals[key] = np.log10(np.abs(transfer_integrals[key]))[mask] #  TODO: log 10
                positions[key] = positions[key][mask]
        else:
            for key in transfer_integrals:
                transfer_integrals[key] = np.abs(transfer_integrals[key])
        if target_domain is not None: # exclude target domain from normalization
            exclude_key = target_domain
            filtered_transfer_integrals = {k: v for k, v in transfer_integrals.items() if k != exclude_key}
            concatenated_target = np.concatenate(list(filtered_transfer_integrals.values()))
        else:
            concatenated_target = np.concatenate(list(transfer_integrals.values()))
        mean = np.mean(concatenated_target)
        std = np.std(concatenated_target)
        print(f'mean: {mean},\t std: {std}')
        return mean, std



# # MBTR_1 as node feature (outside of loop, since same for all molecules in file)
# # each node is represented as a MBTR_1 gaussian corresponding to the respective atom
# num_feat_mbtr = 100
# mol = Atoms(positions=positions[0], numbers=np.squeeze(atomic_nums))
# mbtr = MBTR(
#         species=np.unique(atoms).tolist(),
#         k1={
#             "geometry": {"function": "atomic_number"},
#             "grid": {"min": -3, "max": 10, "n": num_feat_mbtr, "sigma": 1}, # NOTE: to adjust based on atomic numbers
#         },
#         periodic=False,
#         normalization="l2_each",
#         flatten=False
#     )
# mbtr_mol = mbtr.create(mol)
# imap = mbtr.index_to_atomic_number
# smap = {index: ase.data.chemical_symbols[number] for index, number in imap.items()} 
# feat_mbtr = np.zeros((num_nodes, num_feat_mbtr))
# n_elements = len(mbtr.species)
# for i in range(n_elements): 
#     feat_mbtr[np.where(atoms==smap[i])] = mbtr_mol['k1'][i,:]
# feat_mbtr = torch.tensor(feat_mbtr, dtype=torch.double)


# Stratified sampling of data (transfer integrals and positions)
# print(f'Stratification set to: {isinstance(self.stratify, int)}') # might have to remove the last bin
# if isinstance(self.stratify, int):
#     assert self.stratify <= self.debug_size, "The number of strata must be lower or equal to the number of samples"
#     # divide in bins
#     bins = np.linspace(transfer_integrals.min(), transfer_integrals.max(), self.stratify)
#     transfer_integrals_binned = np.digitize(transfer_integrals, bins)
#     stratified_transfer_int = np.asarray([])
#     stratified_pos = np.asarray([]).reshape(0,len(atoms),3)
#     pop_size = np.floor_divide(self.debug_size,self.stratify)
#     # sample equally for each bin (possible the number of sample in each bin should be the same)
#     for stratum in range(1,self.stratify):
#         # debug 
#         # if stratum == 1:
#         #     print('debug')
#         # print(f'Getting {np.sum(transfer_integrals_binned==stratum)} samples for stratum {stratum}.')
#         mask = (transfer_integrals_binned==stratum)
#         stratified_transfer_int = np.concatenate((stratified_transfer_int, transfer_integrals[mask][:pop_size]))
#         stratified_pos = np.concatenate((stratified_pos, positions[mask][:pop_size]))
#     if len(stratified_transfer_int) != self.debug_size: # to fill up debug size, sample randomly with replacement
#         rng = np.random.default_rng(42)
#         sample_size = self.debug_size - len(stratified_transfer_int) 
#         mask = rng.choice(np.arange(len(transfer_integrals)), size=sample_size, replace=False, shuffle=False)
#         stratified_transfer_int = np.concatenate((stratified_transfer_int, transfer_integrals[mask]))
#         stratified_pos = np.concatenate((stratified_pos, positions[mask]))
#     transfer_integrals = stratified_transfer_int
#     positions = stratified_pos
# else:
#     # set data size to debug_size: (do after data has been sifted)
#     print(f'Size of dataset is set to: {self.debug_size}')
#     positions, transfer_integrals = positions[:self.debug_size], transfer_integrals[:self.debug_size]
