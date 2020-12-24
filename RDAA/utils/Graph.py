# coding: utf-8
import networkx as nx
from sklearn.utils import shuffle


class Graph:
    def __init__(self):
        self.g = nx.DiGraph()

    def getGraph(self, edges, weighted, directed):
        for line in edges:
            if directed:
                if not weighted:
                    src, tgt = line
                    self.g.add_edge(src, tgt)
                    self.g[src][tgt]['weight'] = 1.0
                else:
                    src, tgt, weight = line
                    self.g.add_edge(src, tgt)
                    self.g[src][tgt]['weight'] = float(weight)
            else:
                if not weighted:
                    src, tgt = line
                    self.g.add_edge(src, tgt)
                    self.g.add_edge(tgt, src)
                    self.g[src][tgt]['weight'] = 1.0
                    self.g[tgt][src]['weight'] = 1.0
                else:
                    src, tgt, weight = line
                    self.g.add_edge(src, tgt)
                    self.g.add_edge(tgt, src)
                    self.g[src][tgt]['weight'] = float(weight)
                    self.g[tgt][src]['weight'] = float(weight)
        print ('nodes: {}, edges: {}'.format(self.g.number_of_nodes(), self.g.number_of_edges()))

    def remove_self_loop(self):
        self_loop_node = []
        for node in self.g.nodes():
            if node in self.g[node]:
                self.g.remove_edge(node, node)
        for node in self.g.nodes:
            if self.g.in_degree(node) == 0 and self.g.out_degree(node) == 0:
                self_loop_node.append(node)
        if len(self_loop_node) > 0:
            self.g.remove_nodes_from(self_loop_node)
            print ('Remove {} self loop'.format(self_loop_node))
        print ('node: {}, edges: {}').format(len(self.g.nodes), len(self.g.edges))

    def convert_node_labels_to_integers(self, G, first_label=0, ordering="default",
                                        label_attribute=None):
        """Return a copy of the graph G with the nodes relabeled using
        consecutive integers and the mapping dict.

        Parameters
        ----------
        G : graph
        A NetworkX graph

        first_label : int, optional (default=0)
        An integer specifying the starting offset in numbering nodes.
        The new integer labels are numbered first_label, ..., n-1+first_label.

        ordering : string
        "default" : inherit node ordering from G.nodes()
        "sorted"  : inherit node ordering from sorted(G.nodes())
        "increasing degree" : nodes are sorted by increasing degree
        "decreasing degree" : nodes are sorted by decreasing degree

        label_attribute : string, optional (default=None)
        Name of node attribute to store old label.  If None no attribute
        is created.

        Returns
        -------
        G : Graph
        A NetworkX graph
        mapping : dict
        A dict of {node: id}
        Notes
        -----
        Node and edge attribute data are copied to the new (relabeled) graph.
        """
        N = G.number_of_nodes() + first_label
        if ordering == "default":
            mapping = dict(zip(G.nodes(), range(first_label, N)))
        elif ordering == "sorted":
            nlist = sorted(G.nodes())
            mapping = dict(zip(nlist, range(first_label, N)))
        elif ordering == "increasing degree":
            dv_pairs = [(d, n) for (n, d) in G.degree()]
            dv_pairs.sort()  # in-place sort from lowest to highest degree
            mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
        elif ordering == "decreasing degree":
            dv_pairs = [(d, n) for (n, d) in G.degree()]
            dv_pairs.sort()  # in-place sort from lowest to highest degree
            dv_pairs.reverse()
            mapping = dict(zip([n for d, n in dv_pairs], range(first_label, N)))
        else:
            raise nx.NetworkXError('Unknown node ordering: %s' % ordering)
        H = nx.relabel_nodes(G, mapping)
        # create node attribute with the old label
        if label_attribute is not None:
            nx.set_node_attributes(H, {v: k for k, v in mapping.items()}, label_attribute)
        return H, mapping
