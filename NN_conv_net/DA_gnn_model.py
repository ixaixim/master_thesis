# Model for GNN with domain adaptation
# we evaluated a model trained with pentacene (labeled), tetracene and DNTT (both unlabeled)
# test on DNTT
import torch
import torch.nn as nn
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn.conv import NNConv

torch.cuda.empty_cache()

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha*grad_output
        return grad_input, None


class GradientReverse(nn.Module):
    def __init__(self, alpha=1.):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return GradientReversal.apply(x, self.alpha)




class DA_GNN(nn.Module):
    def __init__(self,
    nodefeat_num=3, edgefeat_num=1,
    nodeembed_to=3, edgeembed_to=4,
    num_classes = 2):
        super().__init__()
        self.num_classes = num_classes # number of domains in DA
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

        self.act = nn.ReLU()
        ## Pooling and actuall prediction NN
        self._pooling = [global_mean_pool, global_max_pool] # takes batch.x and batch.batch as args

        # label predictor
        self.label_predictor = nn.Sequential(
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

        # domain predictor
        # NOTE:
        # about complexity: we control the complexity of three components: domain predictor, target predictor, feature extractor
        # obs: making the domain net more complex leads to smaller domain loss
        #       i.e. the model distinguishes better the features -> predictor loss diminishes (predictor loss gets higher)
        #       making domain net simpler leads to dumber distinuish -> small predictor loss 
        # making the feature extractor dumber leads to higher domain loss (which we want)
        # making the predictor smarter leads also to small domain losses (just like feature extractor)
        self.domain = nn.Sequential(
            nn.Linear(in_features=nodeembed_to*len(self._pooling), out_features=16),
            nn.ReLU(inplace = True),
            nn.BatchNorm1d(16),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(in_features=8, out_features=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4),
            # nn.Linear(in_features=6, out_features=3), # for multi-class cross entropy
            nn.Linear(in_features=4, out_features=num_classes), # for multi-class cross entropy
        )
        self.revgrad = GradientReverse()

    def forward(self, batch: Batch, alpha=3):
        ###################
        # feature extractor
        ###################
        node_features, edges, edge_features, batch_vector = \
            batch.x.float(), batch.edge_index, batch.edge_attrs.float(), batch.batch
        ## features embedding
        node_features = self._node_embednorm(
            self._node_embedding(node_features))
        edge_features = self._edge_embednorm(
            self._edge_embedding(edge_features))
        
        # graph convolutions 
        for step in range(len(self.convs)):
            node_features = self.bns[step](self.act(self.convs[step](node_features, edges, edge_features)))

        # pooling
        pooled_graph_nodes = torch.cat([p(node_features, batch_vector) for p in self._pooling], axis=1)

        ###################
        # label predictor
        ###################
        out_1 = self.label_predictor(pooled_graph_nodes)

        # ###################
        # domain predictor
        # ###################
        self.revgrad.alpha = torch.tensor(alpha, requires_grad=False)
        out_2 = self.revgrad(pooled_graph_nodes)
        out_2 = self.domain(out_2)
        # # add out 2 to return

        return out_1, out_2
