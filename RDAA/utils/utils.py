# -*- coding: utf-8 -*-
import torch
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
import scipy.io as sio
import pandas as pd


def softmax(x, axis=None):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def init_snapshoot(data_name, snap_time):
    import os
    snap_root = 'checkpoints/' + data_name.split('/')[-2] + '/' + str(snap_time)
    if not os.path.exists(snap_root):
        os.makedirs(snap_root)
    fout = open(snap_root + '/result.log', 'w')
    return fout, snap_root


def read_baseline_embedding(filename, N):
    emb_df = pd.read_csv(filename, header=None, sep=' ')
    emb_df = emb_df.sort_values(by=0)
    if 'metapath' in filename:
        emb_df = emb_df.iloc[:, :-1]
    emb = np.zeros((N, emb_df.shape[1] - 1), dtype=float)
    for idx, row in emb_df.iterrows():
        emb[int(row[0])] = np.array(row[1:])
    return emb


def load_embedding(fname, method):
    if method in ['deepwalk', 'node2vec', 'line', 'struc2vec']:
        emb_dict = {}
        with open(fname, 'r') as f:
            node, emb_size = list(map(int, f.readline().strip().split(' ')))
            for line in f:
                line = line.strip().split(' ')
                emb_dict[int(line[0])] = np.array(line[1:])
        for i in range(node):
            if i == 0:
                emb = emb_dict[i]
            else:
                emb = np.vstack((emb, emb_dict[i]))
        return emb.astype(np.float64)
    elif method in ['DRNE', 'role']:
        return np.load(fname)
    elif method in ['GraphWave', 'RolX']:
        df = pd.read_csv(fname, sep=',')
        return df.values


def convert_to_csr(filename, fmt='dat'):
    if fmt == 'mat':
        return sio.loadmat(filename)['net']
    # if fmt == 'dat':
        # data_df = pd.read_csv(filename, sep='\t', header=None)
        # data_df.drop_duplicates(inplace=True)
        # row = [int(i[1:])-1 for i in np.array(data_df[0])]
        # col = [int(i[1:])-1 for i in np.array(data_df[1])]
        # data = np.array(data_df[2], dtype=np.float)
        # return csr_matrix((data, (row, col)), dtype=np.float)
    if fmt == 'dblp_old':
        data_df = pd.read_csv(filename, sep='\t', header=None)
        data_df.drop_duplicates(inplace=True)
        row = [int(i[1:]) for i in np.array(data_df[0])]
        col = [int(i[1:]) for i in np.array(data_df[1])]
        data = np.array(data_df[2], dtype=np.float)
        return csr_matrix((data, (row, col)), dtype=np.float)
    if fmt in ['ml-10M', 'dblp', 'wiki', 'dat']:
        data_df = pd.read_csv(filename, sep='\t', header=None)
        data_df.drop_duplicates(inplace=True)
        row = [int(i) - 1 for i in np.array(data_df[0])]
        col = [int(i) - 1 for i in np.array(data_df[1])]
        data = np.array(data_df[2], dtype=np.float)
        return csr_matrix((data, (row, col)), dtype=np.float)


def extend_adj(adj, method='null', implict_k=0):
    N = adj.shape[0]
    M = adj.shape[1]
    if method == 'null':
        n_sim_matrix = csr_matrix((N, N), dtype=np.float)
        m_sim_matrix = csr_matrix((M, M), dtype=np.float)
        pass
    elif method == 'cos_sim':
        n_sim_matrix = cos_sim(adj, implict_k)
        m_sim_matrix = cos_sim(adj.T, implict_k)
    n_sim_matrix = hstack((n_sim_matrix, adj))
    m_sim_matrix = hstack((adj.T, m_sim_matrix))
    return vstack((n_sim_matrix, m_sim_matrix))


def cos_sim(adj, k):
    """calculate the similary between the same type nodes. use the cosin similary:
    sim_{ij} = frac{N_i \cap N_j}{\sqrt{d_i * d_j}}

    Parameter
    ---------
    adj: N*M sparse matrix
    k: threshold of sim_matrix
       Wiki:
       DBLP:
       Movielens:

    Return
    ------
    adj: N*N sparse matrix
    """
    # __import__('pdb').set_trace()
    adj = (csr_matrix(adj) > 0).astype(int)
    com_n = adj.dot(adj.T)
    degree = np.sum(adj, axis=1)
    degree = np.power(degree.dot(degree.T), 0.5)
    # sim_matrix = np.triu(com_n / d, k=1)
    sim_matrix = com_n / degree
    sim_matrix[sim_matrix <= k] = 0
    sim_matrix[np.isnan(sim_matrix)] = 0
    sim_matrix[np.isinf(sim_matrix)] = 0
    print (sim_matrix[sim_matrix > 0].shape[1])
    return sim_matrix


def precision_and_recall(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits/(1.0 * len(ranked_list))
    rec = hits/(1.0 * len(ground_list))
    return pre, rec


def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i+1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0


def read_test_file(filename):
    data_df = pd.read_csv(filename, sep='\t', header=None)
    data_df.drop_duplicates(inplace=True)
    users, items, weight_dict = set(), set(), {}
    for idx, row in data_df.iterrows():
        user = int(row[0] - 1)
        item = int(row[1] - 1)
        weight = float(row[2])
        users.add(user)
        items.add(item)
        if user not in weight_dict:
            weight_dict[user] = {}
        weight_dict[user][item] = weight
    return list(users), list(items), weight_dict


def read_train_file(filename):
    folder_name = '/'.join(filename.split('/')[:-1])
    data_df = pd.read_csv(folder_name + '/train_dup.dat', sep='\t', header=None)
    data_df.drop_duplicates(inplace=True)
    users, items, weight_dict = [], [], {}
    for idx, row in data_df.iterrows():
        user = int(row[0] - 1)
        item = int(row[1] - 1)
        weight = float(row[2])
        users.append(user)
        items.append(item)
        if user not in weight_dict:
            weight_dict[user] = {}
        weight_dict[user][item] = weight
    return users, items, weight_dict


def cat_neighbor(g, embedding, method='null'):
    """concatenate node i neighbor's embedding to node i

    Parameter
    ---------
    g: Graph
    a networkx graph

    embedding: ndarray
    a numpy ndarray which represent nodes embedding

    method: str
    "null": default, use original embedding
    "cat_pos": use positive out edges as neighbor embedding, and concatenate it with original embedding
    "cat_pos_self": like "cat_pos"
    "cat_pos_extend": like "cat_pos", but use in and out edges

    Return
    ------
    emb: ndarray
    the embedding of nodes while concatenating neighbor nodes' embedding

    Notes
    -----
    """
    neighbor_emb = np.zeros_like(embedding)
    if method == 'null':
        return embedding
    elif method == 'cat_neighbor':
        for node in g.nodes:
            neighbor_node = [tgt for src, tgt in g.out_edges(node)] + [src for src, tgt in g.in_edges(node)]
            if len(neighbor_node) == 0:
                assert len(neighbor_node) == 0, 'node {} has no neighbor!!!'.format(node)
                neighbor_emb[node] = embedding[node]
            else:
                neighbor_emb[node] == np.mean(embedding[neighbor_node], axis=0)
        return np.concatenate((embedding, neighbor_emb), axis=1)
    elif method == 'cat_neighbor_attention':
        for node in g.nodes:
            neighbor_node = [tgt for src, tgt in g.out_edges(node)]
            assert len(neighbor_node) != 0, 'node {} has no neighbor!!!'.format(node)
            relevance = np.sum(embedding[node] * embedding[neighbor_node], axis=1)
            relevance = softmax(relevance).reshape(len(neighbor_node), 1)
            neighbor_emb[node] = np.sum(relevance * embedding[neighbor_node], axis=0)
        return np.concatenate((embedding, neighbor_emb), axis=1)
    elif method == 'cat_pos':
        for node in g.nodes:
            neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == 1]
            if len(neighbor_node) == 0:
                continue
            neighbor_emb[node] = np.mean(embedding[neighbor_node], axis=0)
        return np.concatenate((embedding, neighbor_emb), axis=1)
    elif method == 'cat_pos_self':
        for node in g.nodes:
            neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == 1]
            if len(neighbor_node) == 0:
                neighbor_emb[node] = embedding[node]
            else:
                neighbor_emb[node] = np.sum(embedding[neighbor_node], axis=0)
        return np.concatenate((embedding, neighbor_emb), axis=1)
    elif method == 'cat_pos_extend':
        in_neighbor_emb = np.zeros_like(embedding)
        out_neighbor_emb = np.zeros_like(embedding)
        for node in g.nodes:
            out_neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == 1]
            in_neighbor_node = [src for src, tgt in g.in_edges(node) if g[src][node]['sign'] == 1]
            neighbor_node = list(set(out_neighbor_node) | set(in_neighbor_node))
            # if len(in_neighbor_node) == 0:
                # neighbor_emb[node] = embedding[node]
            # else:
                # neighbor_emb[node] = np.mean(embedding[neighbor_node], axis=0)
            # if len(out_neighbor_node) == 0:
                # neighbor_emb[node] = embedding[node]
            # else:
                # neighbor_emb[node] = np.mean(embedding[neighbor_node], axis=0)
            if len(neighbor_node) == 0:
                neighbor_emb[node] = embedding[node]
            else:
                neighbor_emb[node] = np.mean(embedding[neighbor_node], axis=0)
        return np.concatenate((embedding, neighbor_emb), axis=1)
    elif method == 'cat_pos_attention':
        # def softmax(x):
            # x = x - np.max(x)
            # exp_x = np.exp(x)
            # return exp_x / np.sum(exp_x)
        for node in g.nodes:
            neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[src][tgt]['sign'] == 1]
            if len(neighbor_node) == 0:
                # neighbor_emb[node] = embedding[node]
                continue
            else:
                # __import__('pdb').set_trace()
                relevance = np.sum(embedding[node] * embedding[neighbor_node], axis=1)
                # relevance = relevance / np.sum(relevance)
                relevance = softmax(relevance)
                # relevance = np.ones(embedding.shape[1], dtype=np.float)
                neighbor_emb[node] = np.sum(relevance * embedding[neighbor_node], axis=0)
        return np.concatenate((embedding, neighbor_emb), axis=1)
    elif method == 'cat_pos_degree_attention':
        for node in g.nodes:
            neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[src][tgt]['sign'] == 1]
            if len(neighbor_node) == 0:
                neighbor_emb[node] = embedding[node]
                continue
            else:
                # relevance = np.sum(embedding[node] * embedding[neighbor_node], axis=1).reshape(len(neighbor_node), 1)
                relevance = np.array([1.0 * g.degree(i) for i in neighbor_node], dtype=np.float)
                relevance = softmax(relevance).reshape(len(neighbor_node), 1)
                # relevance = (relevance / np.sum(relevance)).reshape(len(neighbor_node), 1)
                # relevance = np.ones(embedding.shape[1], dtype=np.float)
                neighbor_emb[node] = np.sum(relevance * embedding[neighbor_node], axis=0)
        return np.concatenate((embedding, neighbor_emb), axis=1)
    elif method == 'cat_pos_neg':
        pos_neighbor_emb = np.zeros_like(embedding)
        neg_neighbor_emb = np.zeros_like(embedding)
        for node in g.nodes:
            pos_neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == 1]
            neg_neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == -1]
            if len(pos_neighbor_node) == 0:
                pos_neighbor_emb[node] = embedding[node]
            else:
                pos_neighbor_emb[node] = np.mean(embedding[pos_neighbor_node], axis=0)
            if len(neg_neighbor_node) == 0:
                neg_neighbor_emb[node] = embedding[node]
            else:
                neg_neighbor_emb[node] = -1.0 * np.mean(embedding[neg_neighbor_node], axis=0)
        return np.concatenate((embedding, pos_neighbor_emb, neg_neighbor_emb), axis=1)
    elif method == 'cat_pos_neg_extend':
        in_pos_neighbor_emb = np.zeros_like(embedding)
        in_neg_neighbor_emb = np.zeros_like(embedding)
        out_pos_neighbor_emb = np.zeros_like(embedding)
        out_neg_neighbor_emb = np.zeros_like(embedding)
        for node in g.nodes:
            in_pos_neighbor_node = [src for src, tgt in g.in_edges(node) if g[src][tgt]['sign'] == 1]
            out_pos_neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[src][tgt]['sign'] == 1]
            in_neg_neighbor_node = [src for src, tgt in g.in_edges(node) if g[src][tgt]['sign'] == -1]
            out_neg_neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[src][tgt]['sign'] == -1]
            if len(in_pos_neighbor_node) == 0:
                in_pos_neighbor_emb[node] = embedding[node]
            else:
                in_pos_neighbor_emb[node] = np.mean(embedding[in_pos_neighbor_node], axis=0)
            if len(in_neg_neighbor_node) == 0:
                in_neg_neighbor_emb[node] = embedding[node]
            else:
                in_neg_neighbor_emb[node] = np.mean(embedding[in_neg_neighbor_node], axis=0)
            if len(out_neg_neighbor_node) == 0:
                out_neg_neighbor_emb[node] = embedding[node]
            else:
                out_neg_neighbor_emb[node] = np.mean(embedding[out_neg_neighbor_node], axis=0)
            if len(out_pos_neighbor_node) == 0:
                out_pos_neighbor_emb[node] = embedding[node]
            else:
                out_pos_neighbor_emb[node] = np.mean(embedding[out_pos_neighbor_node], axis=0)
        return np.concatenate((
            embedding, out_pos_neighbor_emb, in_pos_neighbor_emb, out_neg_neighbor_emb, in_neg_neighbor_emb), axis=1)
    elif method == 'cat_pos_neg_attention':
        pos_neighbor_emb = np.zeros_like(embedding)
        neg_neighbor_emb = np.zeros_like(embedding)
        for node in g.nodes:
            pos_neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == 1]
            neg_neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == -1]
            if len(pos_neighbor_node) == 0:
                pos_neighbor_emb[node] = embedding[node]
            else:
                relevance = np.sum(embedding[node] * embedding[pos_neighbor_node], axis=1).reshape(1, -1)
                relevance = softmax(relevance).reshape(-1, 1)
                pos_neighbor_emb[node] = np.sum(relevance * embedding[pos_neighbor_node], axis=0)
            if len(neg_neighbor_node) == 0:
                neg_neighbor_emb[node] = embedding[node]
            else:
                neg_neighbor_emb[node] = -1.0 * np.mean(embedding[neg_neighbor_node], axis=0)
        return np.concatenate((embedding, pos_neighbor_emb, neg_neighbor_emb), axis=1)
    elif method == 'cat_neg':
        for node in g.nodes:
            neighbor_node = [tgt for src, tgt in g.out_edges(node) if g[node][tgt]['sign'] == -1]
            if len(neighbor_node) == 0:
                neighbor_emb[node] = embedding[node]
            else:
                neighbor_emb[node] = np.sum(embedding[neighbor_node], axis=0)
        return np.concatenate((embedding, neighbor_emb), axis=1)
    else:
        raise Exception('no method named: ' + method)
