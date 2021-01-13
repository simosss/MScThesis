class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.R = torch.empty(32, 32)
        self.D = torch.empty(32, 32)
        nn.init.xavier_uniform_(self.R, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.D, gain=nn.init.calculate_gain('relu'))

    def encode(self):
        x = self.conv1(data.x, data.train_pos_edge_index)
        x = x.relu()
        x = self.conv2(x, data.train_pos_edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_decagon(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = z[pos_edge_index[0]]@self.D@self.R@self.D@z[pos_edge_index[1]].t()
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
