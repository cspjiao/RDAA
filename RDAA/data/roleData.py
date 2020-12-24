# -*- coding: utf-8 -*-
from torch.utils import data
import pickle
import os
from utils import utils


class roleData(data.Dataset):

    def __init__(self, graph):
        self.nodes = list(graph.g.nodes)
        # self.features = graph.features
        # self.adj = graph.get_adjmatrix()

    def __getitem__(self, index):
        # node_feature = self.features[self.nodes[index]].toarray().flatten()
        # neighbor = self.adj[self.nodes[index]].toarray()
        # neighbors = list(self.G.g.neighbors(self.nodes[index]))
        # neighbor_feature = self.G.features[neighbors].toarray()
        # return [node_feature, neighbor_feature]
        # return node_feature, neighbor
        return self.nodes[index]

    def __len__(self):
        return len(self.nodes)
