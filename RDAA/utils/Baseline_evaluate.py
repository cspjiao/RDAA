# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import utils


class Baseline_evaluate(object):
    def __init__(self, Graph, config):
        self.G = Graph
        self.g = Graph.g
        self.config = config
        self.N = self.G.N + self.G.M
        self.user_num = self.G.N
        self.item_num = self.G.M
        # self.adj_matrix = hstack((csr_matrix((self.G.N, self.G.N), dtype=np.float), self.G.ori_adj)).tocsr()
        self.adj_matrix = self.G.ori_adj
        self.data_root = '/'.join(config.filename.split('/')[:-1])
        self.un_g = Graph.g.to_undirected()

    def link_prediction_similarity(self, edge, method):
        if method == 'CN':
            return len(set(self.un_g[edge[0]]) & set(self.un_g[edge[1]]))
        elif method == 'JC':
            res = nx.jaccard_coefficent(self.un_g, [edge])
        elif method == 'AA':
            res = nx.adamic_adar_index(self.un_g, [edge])
        elif method == 'PA':
            res = nx.preferential_attachment(self.un_g, [edge])
        elif method == 'Katz':
            res = 0
        else:
            raise Warning('No Method named {}!!!!!!!!'.format(method))
        for _, _, p in res:
            return p

    def Karz_similarity():
        pass

    def link_prediction_baseline(self, method=['CN', 'JC', 'AA', 'Katz', 'PA']):
        sim_matrix = np.zeros((self.user_num, self.item_num), dtype=float)
        for algo in method:
            for i in xrange(self.user_num):
                for j in xrange(self.item_num):
                    sim_matrix[i][j] = self.link_prediction_similarity((i, j + self.user_num), method=algo)
            __import__('pdb').set_trace()
            sim_matrix = utils.softmax(sim_matrix, axis=1)
            sim_matrix[sim_matrix < 0.5] = 0
            sim_matrix[sim_matrix >= 0.5] = 1
            self.baseline_metrics(sim_matrix)

    def baseline_metrics(self, sim_matrix):
            def get_train_test_set(filename):
                df = pd.read_csv(self.data_root + filename, sep='\t', header=None)
                src = [int(i)-1 for i in np.array(df[0])]
                tgt = [int(i)-1 for i in np.array(df[1])]
                data = np.array(df[2])
                return zip(src, tgt), data
            test_edges, y_test = get_train_test_set('/case_test_dup.dat')
            y_pred = []
            for src, tgt in test_edges:
                y_pred.append(sim_matrix[src][tgt])
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            eval_dict = {'auc': metrics.auc(fpr, tpr),
                         'average_precision': metrics.average_precision_score(y_test, y_pred),
                         'f1': metrics.f1_score(y_test, y_pred),
                         'f1-micro': metrics.f1_score(y_test, y_pred, average='micro'),
                         'f1-macro': metrics.f1_score(y_test, y_pred, average='macro')}
            print ("link_prediction auc: {:.3f}, precison: {:.3}, f1: {:.3f}, f1-micro: {:.3f}, f1-macro: {:.3f}" \
                .format(eval_dict['auc'], eval_dict['average_precision'], eval_dict['f1'],
                        eval_dict['f1-micro'], eval_dict['f1-macro']))
            return eval_dict

    def get_link_embedding(self, embedding, src, tgt, method):
        if method == 'concatenate':
            return np.concatenate((embedding[src, :], embedding[tgt, :]), axis=1)
        if method == 'concatenate_direct':
            return np.concatenate(
                (embedding[src, :self.config.dimension], embedding[tgt, self.config.dimension:]), axis=1)
        if method == '-':
            return embedding[tgt, :] - embedding[src, :]
        if method == '-_relu':
            emb = embedding[tgt, :] - embedding[src, :]
            return (abs(emb) + emb)/2
        if method == 'average':
            return (embedding[src, :] + embedding[tgt, :]) / 2
        if method == 'hadamard':
            return (embedding[src, :] * embedding[tgt, :])
        if method == 'l1':
            return np.abs(embedding[src, :] - embedding[tgt, :])
        if method == 'l2':
            return np.power(embedding[src, :] - embedding[tgt, :], 2.0)

    def check_reconstruction(self, embedding, check_index):
        def get_precisionK(embedding, max_index):
            print ("get precisionK...")
            similarity = self.getSimilarity(embedding)[:self.G.N, :][:, self.G.N:].reshape(-1)
            sortedInd = np.argsort(similarity)
            cur = 0
            count = 0
            precisionK = []
            sortedInd = sortedInd[::-1]
            for ind in sortedInd:
                x = ind / self.G.M
                y = ind % self.G.M
                count += 1
                if (self.adj_matrix[x].toarray()[0][y] == 1 or x == y):
                    cur += 1
                precisionK.append(1.0 * cur / count)
                if count > max_index:
                    break
            return precisionK

        precisionK = get_precisionK(embedding, np.max(check_index))
        ret = []
        for index in check_index:
            print ("precisonK[%d] %.2f" % (index, precisionK[index - 1]))
            ret.append(precisionK[index - 1])
        return ret

    def getSimilarity(self, result):
        return np.dot(result, result.T)

    def link_prediction_by_lr(self, embedding, method='concatenate'):
        def get_train_test_set(filename, embedding, method):
            df = pd.read_csv(self.data_root + filename, sep='\t', header=None)
            src = [int(i)-1 for i in np.array(df[0])]
            tgt = [int(i)-1 + self.G.N for i in np.array(df[1])]
            data = np.array(df[2])
            return self.get_link_embedding(embedding, src, tgt, method), data
        x_train, y_train = get_train_test_set('/case_train_dup.dat', embedding=embedding, method=method)
        x_test, y_test = get_train_test_set('/case_test_dup.dat', embedding=embedding, method=method)
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        # __import__('pdb').set_trace()
        y_score = clf.predict_proba(x_test)[:, -1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        eval_dict = {'auc': metrics.auc(fpr, tpr),
                     'average_precision': metrics.average_precision_score(y_test, y_score),
                     'f1': metrics.f1_score(y_test, y_pred),
                     'f1-micro': metrics.f1_score(y_test, y_pred, average='micro'),
                     'f1-macro': metrics.f1_score(y_test, y_pred, average='macro')}
        print ("link_prediction auc: {:.4f}, precison: {:.4f}, f1: {:.4f}, f1-micro: {:.4f}, f1-macro: {:.4f}" \
            .format(eval_dict['auc'], eval_dict['average_precision'], eval_dict['f1'],
                    eval_dict['f1-micro'], eval_dict['f1-macro']))
        return eval_dict

    def top_N_speedup(self, embedding, top_n):
        # __import__('pdb').set_trace()
        embedding_u = embedding[:self.G.N, :]
        embedding_v = embedding[self.G.N:, :]
        # test_file = 'data/dataset/dblp/rating_test.dat'
        test_file = None
        test_file = self.data_root + '/test_dup.dat'
        if test_file is None:
            users, items, weight = zip(*self.G.test_edges)
            test_rate = {}
            users = list(set(users))
            items = list(set(np.array(items) - self.G.N))
            for u, i, w in self.G.test_edges:
                if u not in test_rate:
                    test_rate[u] = {}
                test_rate[u][i - self.G.N] = w
        else:
            users, items, test_rate = utils.read_test_file(test_file)
        recommend_array = embedding_u.dot(embedding_v.T)
        precison_list = []
        recall_list = []
        ap_list = []
        for u in users:
            tmp_r = sorted(zip(items, list(recommend_array[u, :][items])), lambda x, y: cmp(x[1], y[1]), reverse=True)[0:min(len(items), top_n)]
            tmp_t = sorted(test_rate[u].items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[0:min(len(test_rate[u]), len(test_rate[u]))]
            tmp_r_list = []
            tmp_t_list = []
            for (item, rate) in tmp_r:
                tmp_r_list.append(item)
            for (item, rate) in tmp_t:
                tmp_t_list.append(item)
            pre, rec = utils.precision_and_recall(tmp_r_list, tmp_t_list)
            ap = utils.AP(tmp_r_list, tmp_t_list)
            ap_list.append(ap)
            precison_list.append(pre)
            recall_list.append(rec)
        precison = sum(precison_list) / len(precison_list)
        recall = sum(recall_list) / len(recall_list)
        f1 = 2 * precison * recall / (precison + recall)
        map_val = sum(ap_list) / len(ap_list)
        print ("recommendation metrics\nF1: {:.4f}, Map: {:.4f}".format(f1, map_val))
