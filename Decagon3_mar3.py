import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, RGCNConv
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.data import NeighborSampler

# from tqdm import tqdm
# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------
from dataset import Dec

# Include  the name of the project folder name="...." and the directory of it root="....",
# dataset = Dec(name="...", root="...")
dataset = Dec()
data = dataset[0]

data.train_pos_edge_index = data.edge_index
data.val_pos_edge_index = data.edge_index
data.test_pos_edge_index = data.edge_index

# ------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------

# returns a negative index with exactly one negative edge for each positive
def neg_sampling(edge_index, num_nodes):
    struc_neg_sampl = pyg_utils.structured_negative_sampling(edge_index, num_nodes)
    i,j,k = struc_neg_sampl
    i = i.tolist()
    k = k.tolist()
    neg_edge_index = [i,k]
    neg_edge_index = torch.tensor(neg_edge_index)
    return neg_edge_index

# returns a tensor of labels 1 or 0
def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

#------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------

class Net(torch.nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        num_layers = 2

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(dataset.num_features, 64))
        self.convs.append(GCNConv(64, 32))
        self.R = torch.empty(32, 32)
        self.D = torch.empty(32, 32)
        nn.init.xavier_uniform_(self.R, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.D, gain=nn.init.calculate_gain('relu'))

    #     def __init__(self):
    #         super(Net, self).__init__()
    #         self.conv1 = GCNConv(dataset.num_features, 64)
    #         self.conv2 = GCNConv(64, 32)
    #         self.R = torch.empty(32, 32)
    #         self.D = torch.empty(32, 32)
    #         nn.init.xavier_uniform_(self.R, gain=nn.init.calculate_gain('relu'))
    #         nn.init.xavier_uniform_(self.D, gain=nn.init.calculate_gain('relu'))

    def encode(self, x, adjs):
        edge_index1, _, size1 = adjs[0]
        edge_index2, _, size2 = adjs[1]
        x = self.convs[0](x, edge_index1)
        x = x.relu()
        x = self.convs[1](x, edge_index2)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_decagon(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] @ self.D @ self.R @ self.D @ z[edge_index[1]].t()).sum(dim=1)
        # logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def inference(self, x_all):

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(2):
            xs = []
            nids = []
            for batch_s, n_id, adj in subgraph_loader:
                edge_index, _, size = adj
                x = x_all[n_id]
                x = self.convs[i](x, edge_index)
                if i != 1:
                    x = F.relu(x)
                xs.append(x)
                nids.append(n_id)

            x_all = torch.cat(xs, dim=0)

        return x_all, nids

    # def decode_all(self, z):
    #     prob_adj = z @ z.t()
    #     return (prob_adj > 0).nonzero(as_tuple=False).t()

#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------

def train():

    model.train()

    for batch_size, n_id, adjs in train_loader:
        z = model.encode(data.x[n_id], adjs)
        # z = torch.tensor(z, dtype=torch.float64)
        pos_edge_index, _, _ = adjs[0]
        neg_edge_index = neg_sampling(pos_edge_index, len(n_id))
        optimizer.zero_grad()

        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        loss.backward()
        optimizer.step()

    return loss


@torch.no_grad()
def test():
    model.eval()
    perfs = []
    z, nid = model.inference(data.x) # remember to fix this for actual validation set
    pos_edge_index = data.val_pos_edge_index
    neg_edge_index = neg_sampling(pos_edge_index, len(data.edge_type)) # remember to fix this edge_type len for validatio set
    link_logits = model.decode(z, pos_edge_index, neg_edge_index)
    link_probs = link_logits.sigmoid()
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    perfs.append(roc_auc_score(link_labels, link_probs))
    return perfs

#--------------------------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------------------------

train_loader = NeighborSampler(data.train_pos_edge_index, batch_size=2048, shuffle=True,sizes=[-1,-1])
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1], batch_size=128, shuffle=False)

model = Net()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

#----------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------
best_val_perf = test_perf = 0
for epoch in range(1, 3):
    train_loss = train()
    #print(epoch, train_loss)
    val_perf = test()
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}'
    print(log.format(epoch, train_loss, val_perf[0]))
