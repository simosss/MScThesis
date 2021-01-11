import os
import os.path as osp

import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
from torch_geometric.data import Data



class Dec(InMemoryDataset):

    path= '/Users/simos/Desktop/'

    def __init__(self, root=path, name='Decagon', transform=None, pre_transform=None):
        self.name = name
        self.root = root
        super(Dec, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name,'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def num_relations(self):
        return self.data.edge_type.max().item() + 1

    @property
    def raw_file_names(self):
        return ['decagon_sample.csv', 'bio-decagon-mono.csv']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):

        # READ
        #read sample (complete) dataset into file and mono se dataset
        file =  pd.read_csv(self.raw_paths[0])
        mono = pd.read_csv(self.raw_paths[1])


        # CREATE NECCESARY DICTIONARIES
        #Create dictionaries that map every node and every relation to an integer respectively
        node1 = file['node1'].tolist()
        node2 = file['node2'].tolist()
        rel = file['relation'].tolist()

        r = set(rel)
        n1 = set(node1)
        n2 = set(node2)
        nodes = n1.union(n2)
        num_nodes = len(nodes)
        num_relations = len(r)
        rel_dict = {i:val for val, i in enumerate(sorted(r, reverse = True))}
        nodes_dict = {i:val for val, i in enumerate(sorted(nodes, reverse = True))}
        inv_nodes_dict = {v: k for k, v in nodes_dict.items()}

        # Recreate the original dataset with the integer values for nodes and relations
        file['node1'] = file['node1'].map(nodes_dict)
        file['node2'] = file['node2'].map(nodes_dict)
        file['relation'] = file['relation'].map(rel_dict)

        # create a dictionary to link mono side effects to their names
        mono_sename_dict = {}
        for (se, se_name) in zip(mono['Individual Side Effect'],mono['Side Effect Name']):
            mono_sename_dict[se] = se_name
        num_features = len(mono_sename_dict)

        # create a dictionary to link drugs to their mono se
        drug_se_dict = defaultdict(set)
        for (drug, se) in zip(mono['STITCH'],mono['Individual Side Effect']):
            drug_se_dict[drug].add(se)

        # a list of all mono side effects
        side_effects = sorted(list((mono_sename_dict.keys())))

        #create a list of lists holding the feature vectors for every drug (which has mono se)
        drug_features={}
        for drug in drug_se_dict:
            vector=[]
            for se in side_effects:
                if se in drug_se_dict[drug]:
                    vector.append(1)
                else:
                    vector.append(0)
            drug_features[drug]=vector


        # A dictionary that for every node holds its feature vector. (Based on mono se if applicable or else, random)
        features={}
        for node in nodes:
            if node in drug_se_dict:
                features[node]= drug_features[node]
            else:
                features[node]= list(np.random.randint(2, size=num_features))
        print("Finished with creation of dictionaries")

        # Create the tensor that holds the features for all the nodes
        li = [ [] for _ in range(num_nodes)]
        for i in range(num_nodes):
            li[i] = features[inv_nodes_dict[i]]
        x = torch.tensor(li, dtype=torch.float)
        print("Finished creating features")

        #Create the edge index, edge type from the dataset
        edge_list=[]
        edge_type=[]
        for i,j,k in zip(file['node1'],file['node2'],file['relation']):
            edge_list.append([i,j])
            edge_type.append(k)
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        #edge_index = pyg_utils.to_undirected(edge_index,n_nodes)
        edge_type = torch.tensor(edge_type, dtype=torch.long)

        data = Data(edge_index=edge_index)
        data.num_nodes = num_nodes
        #x = torch.rand(data.num_nodes, num_features, dtype=torch.float)
        data.x = x
        data.edge_type = edge_type
        #data.nodes=nodes

        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return '{}{}()'.format(self.name.upper(), self.__class__.__name__)
