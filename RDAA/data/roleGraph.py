# coding: utf-8
import networkx as nx
from sklearn.utils import shuffle
from utils import Graph
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


class roleGraph(Graph):
    def __init__(self, config, sep=' ', remove_self_loop=False,
                 train_with_only_trainset=False):
        super(roleGraph, self).__init__()
        self.graph_dir = '../dataset/{}.edge'.format(config.dataset)
        self.feature_dir = '../cache/features/{}_features.csv'.format(config.dataset)

        # construct Graph
        graph_df = pd.read_csv(self.graph_dir, sep=sep, header=None, dtype=int)
        graph_df.drop_duplicates(inplace=True)
        edgelist = graph_df.values.tolist()
        src, tgt = list(set(zip(*edgelist)))
        nodes = list(set(src) | set(tgt))
        assert len(nodes) == max(nodes) + 1
        for node in range(max(nodes) + 1):
            self.g.add_node(node)
        self.getGraph(edgelist, weighted=config.weighted, directed=config.directed)
        if remove_self_loop:
            self.remove_self_loop()
        # if config.split_ratio == 1.0:
        # self.test_edges = self.train_edges
        # if train_with_only_trainset:
        # self.all_edges = self.train_edges
        # self.g.remove_edges_from(list(self.g.edges))
        #     self.getGraph(weighted=self.weighted, directed=self.directed)

        # read features
        feature_pd = pd.read_csv(self.feature_dir, sep=',')
        # self.features=csr_matrix(np.load('data/RolX/output/test.npy'), dtype=np.float32)
        self.features = csr_matrix(feature_pd.values, dtype=np.float32)

    def get_adjmatrix(self):
        return nx.to_scipy_sparse_matrix(self.g,
                                         nodelist=list(range(self.g.number_of_nodes())),
                                         weight='weight', format='csr')
