# -*- coding: utf-8 -*-
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
import pandas as pd
from utils import utils
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer


class Task(object):
    def __init__(self, Graph, config):
        self.G = Graph
        self.g = Graph.g
        self.config = config
        self.label_file = '../dataset/{}.lbl'.format(self.config.dataset)
        self.labels_np = np.loadtxt(self.label_file, dtype=np.int)


    def _classfication(self, embedding, split_ratio=0.8):
        labels_np = shuffle(self.labels_np)
        nodes = labels_np[:, 0]
        labels = labels_np[:, 1]
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        train_size = int(labels_np.shape[0] * split_ratio)
        features = embedding[nodes]
        train_x = features[:train_size, :]
        train_y = labels[:train_size, :]
        test_x = features[train_size:, :]
        test_y = labels[train_size:, :]
        clf = OneVsRestClassifier(LogisticRegression(class_weight='balanced', solver='liblinear', n_jobs=-1))
        # clf = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=100)
        clf.fit(train_x, train_y)
        y_pred = clf.predict_proba(test_x)
        y_pred = lb.transform(np.argmax(y_pred, 1))
        acc = np.sum(np.argmax(y_pred, 1) == np.argmax(test_y, 1))/len(y_pred)
        # fpr, tpr, thresholds = metrics.roc_curve(test_y, y_score)
        eval_dict = {'acc': acc,
                     'f1-micro': metrics.f1_score(np.argmax(test_y, 1), np.argmax(y_pred, 1), average='micro'),
                     'f1-macro': metrics.f1_score(np.argmax(test_y, 1), np.argmax(y_pred, 1), average='macro')}
        return eval_dict

    def classfication(self, embedding, split_ratio=0.8, loop=1):
        eval_dict = {'acc': 0.0, 'f1-micro': 0.0, 'f1-macro': 0.0}
        for _ in range(loop):
            tmp_dict = self._classfication(embedding, split_ratio)
            for key in tmp_dict.keys():
                eval_dict[key] += tmp_dict[key]
        for key in tmp_dict.keys():
            eval_dict[key] = round((1.0 * eval_dict[key]) / loop, 4)
        print ('split_ratio: {}'.format(split_ratio))
        print (eval_dict)
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
        clf = LogisticRegression(class_weight='balanced', solver='liblinear')
        # clf = LogisticRegression(class_weight='balanced', solver='lbfgs', max_iter=100)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        # __import__('pdb').set_trace()
        y_score = clf.predict_proba(x_test)[:, -1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
        eval_dict = {'auc': metrics.auc(fpr, tpr),
                     'pr': metrics.average_precision_score(y_test, y_score),
                     'f1': metrics.f1_score(y_test, y_pred),
                     'f1-micro': metrics.f1_score(y_test, y_pred, average='micro'),
                     'f1-macro': metrics.f1_score(y_test, y_pred, average='macro')}
        print ("link_prediction auc: {:.4f}, precison: {:.4}, f1: {:.4f}, f1-micro: {:.4f}, f1-macro: {:.4f}" \
            .format(eval_dict['auc'], eval_dict['pr'], eval_dict['f1'],
                    eval_dict['f1-micro'], eval_dict['f1-macro']))
        return eval_dict

    def top_N(self, embedding, top_n):
        # __import__('pdb').set_trace()
        embedding_u = embedding[:self.G.N, :]
        embedding_v = embedding[self.G.N:, :]
        user, item, weight = zip(*self.G.test_edges)
        test_rate = {}
        user = list(set(user))
        item = list(np.array(item) - self.G.N)
        for u, i, w in self.G.test_edges:
            if u in test_rate:
                test_rate[u][i - self.G.N] = w
            else:
                test_rate[u] = {i - self.G.N: w}
        recommend_dict = {}
        for u in user:
            recommend_dict[u] = {}
            for v in item:
                U = embedding_u[u]
                V = embedding_v[v]
                pre = U.dot(V.T)
                recommend_dict[u][v] = float(pre)
        precison_list = []
        recall_list = []
        ap_list = []
        ndcg_list = []
        rr_list = []
        for u in user:
            tmp_r = sorted(recommend_dict[u].items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[0:min(len(recommend_dict[u]), top_n)]
            tmp_t = sorted(test_rate[u].items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[0:min(len(test_rate[u]), len(test_rate[u]))]
            tmp_r_list = []
            tmp_t_list = []
            for (item, rate) in tmp_r:
                tmp_r_list.append(item)
            for (item, rate) in tmp_t:
                tmp_t_list.append(item)
            pre, rec = utils.precision_and_recall(tmp_r_list, tmp_t_list)
            precison_list.append(pre)
            recall_list.append(rec)
        precison = sum(precison_list) / len(precison_list)
        recall = sum(recall_list) / len(recall_list)
        f1 = 2 * precison * recall / (precison + recall)
        print("recommendation metrics\nF1: {:.4f},".format(f1))

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
        recommend_dict = {}
        recommend_array = embedding_u.dot(embedding_v.T)
        precison_list = []
        recall_list = []
        ap_list = []
        ndcg_list = []
        rr_list = []
        for u in users:
            tmp_r = sorted(zip(items, list(recommend_array[u, :][items])), key=lambda x: x[1], reverse=True)[0:min(len(items), top_n)]
            # tmp_r = sorted(recommend_dict[u].items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[0:min(len(recommend_dict[u]), top_n)]
            tmp_t = sorted(test_rate[u].items(), key=lambda x: x[1], reverse=True)[0:min(len(test_rate[u]), len(test_rate[u]))]
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
        print("recommendation metrics\nF1: {:.4f}, Map: {:.4f}".format(f1, map_val))
        return {'f1': f1, 'map': map_val}
