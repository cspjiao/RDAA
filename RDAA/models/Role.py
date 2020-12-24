# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .BasicModule import BasicModule


class RDAA(BasicModule):
    '''
    BiAE: use extend adj and Auto-Encoder
    '''

    def __init__(self, config, graph):
        super(RDAA, self).__init__()
        self.config = config
        self.model_name = 'Role'
        self.device = torch.device('cuda:{}'.format(config.device) if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(0.5)
        self.encoder_list = nn.ModuleList([
            nn.Linear(self.config.struct[i], self.config.struct[i + 1])
            for i in range(len(self.config.struct) - 1)])
        self.config.struct.reverse()
        self.decoder_list = nn.ModuleList([
            nn.Linear(self.config.struct[i], self.config.struct[i + 1])
            for i in range(len(self.config.struct) - 1)])
        self.config.struct.reverse()
        # self.all_features = features
        # self.nodes_num, self.feature_num = features.size()
        self.query = nn.Parameter(torch.Tensor(1, 2 * self.config.struct[-1]))
        self.init_model_weight()
        self.G = graph
        # self.init_model_weight_RMSprop()

    def init_model_weight(self):
        for i in range(len(self.config.struct) - 1):
            nn.init.xavier_uniform_(self.encoder_list[i].weight)
            nn.init.uniform_(self.encoder_list[i].bias, a=-0.5, b=0.5)
            nn.init.xavier_uniform_(self.decoder_list[i].weight)
            nn.init.uniform_(self.decoder_list[i].bias, a=-0.5, b=0.5)
        nn.init.uniform_(self.query, a=-0.5, b=0.5)

    def init_model_weight_RMSprop(self):
        for i in range(len(self.config.struct) - 1):
            nn.init.normal_(self.encoder_list[i].weight)
            nn.init.uniform_(self.encoder_list[i].bias, a=-0.5, b=0.5)
            nn.init.normal_(self.decoder_list[i].weight)
            nn.init.uniform_(self.decoder_list[i].bias, a=-0.5, b=0.5)
        nn.init.uniform_(self.query, a=-0.5, b=0.5)

    def encoder(self, X):
        for i in range(len(self.config.struct) - 1):
            X = torch.tanh(self.encoder_list[i](X))
        return X

    def decoder(self, X):
        for i in range(len(self.config.struct) - 1):
            X = torch.tanh(self.decoder_list[i](X))
        return X

    def get_context_enc(self, nodes):
        nodes = nodes.cpu().numpy()
        flag = True
        for node in nodes:
            neighbors = list(self.G.g.neighbors(node))
            neighbors_feature = torch.Tensor(self.G.features[neighbors].toarray()).to(device=self.device,
                                                                                      dtype=torch.float32)
            neighbors_enc = self.encoder(neighbors_feature)
            node_feature = torch.Tensor(self.G.features[node].toarray()).to(device=self.device, dtype=torch.float32)
            _node_enc = self.encoder(node_feature)
            node_enc_extend = _node_enc * torch.ones_like(neighbors_enc)
            _cat_node_nei = torch.cat((node_enc_extend, neighbors_enc), dim=1)
            alpha = F.softmax(F.relu(torch.matmul(_cat_node_nei, self.query.t())), dim=0)
            # alpha = F.softmax((node_enc_extend * neighbors_enc).sum(1, keepdim=True), dim=0)
            _context_enc = (alpha * neighbors_enc).sum(0, keepdim=True)
            if flag:
                flag = False
                node_enc = _node_enc
                context_enc = _context_enc
            else:
                node_enc = torch.cat((node_enc, _node_enc), dim=0)
                context_enc = torch.cat((context_enc, _context_enc), dim=0)
        return node_enc, context_enc

    def get_2nd_loss(self, X, new_X):
        B = X * (self.config.beta - 1) + 1
        return torch.pow((new_X - X) * B, 2).sum()

    def get_1st_loss(self, mu_i, sigma_i, mu_j, sigma_j, mu_k, sigma_k):
        e_ij = self._wasserstein_distice(mu_i, sigma_i, mu_j, sigma_j)
        e_ik = self._wasserstein_distice(mu_i, sigma_i, mu_k, sigma_k)
        # return -1.0 * (X * (torch.matmul(encoder, encoder.transpose(0, 1)))).sum()
        # return (torch.pow((s_encoder - t_encoder), 2) * S.view(-1, 1)).sum()
        return (e_ij + torch.exp(-1 * torch.pow(e_ik, 0.5))).sum()

    def _wasserstein_distice(self, mu1, sigma1, mu2, sigma2):
        return (torch.norm(mu1 - mu2, p=2, dim=1) + torch.norm(sigma1 - sigma2, p=2, dim=1))

    def forward(self, data):
        nodes, node_feature, neighbor_mask = data
        # encoder-decoder
        # node_enc = self.encoder(node_feature)
        # node_dec = self.decoder(node_enc)

        node_enc, context_enc = self.get_context_enc(nodes)
        _role_loss = torch.pow(torch.norm(node_enc - context_enc), 2)

        node_dec = self.decoder(node_enc)
        _enc_dec_loss = self.get_2nd_loss(node_feature, node_dec)
        return _enc_dec_loss + self.config.alpha * _role_loss

    def get_embedding(self, features):
        enc = self.encoder(features)
        return enc

    def save_embedding(self, emb, fout):
        np.save(fout, emb)
        return None


class RDAE(BasicModule):
    '''
    BiAE: use extend adj and Auto-Encoder
    '''

    def __init__(self, config, graph):
        super(RDAE, self).__init__()
        self.config = config
        self.model_name = 'Role'
        self.device = torch.device('cuda:{}'.format(config.device) if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(0.5)
        self.encoder_list = nn.ModuleList([
            nn.Linear(self.config.struct[i], self.config.struct[i + 1])
            for i in range(len(self.config.struct) - 1)])
        self.config.struct.reverse()
        self.decoder_list = nn.ModuleList([
            nn.Linear(self.config.struct[i], self.config.struct[i + 1])
            for i in range(len(self.config.struct) - 1)])
        self.config.struct.reverse()
        # self.all_features = features
        # self.nodes_num, self.feature_num = features.size()
        self.query = nn.Parameter(torch.Tensor(1, 2 * self.config.struct[-1]))
        self.init_model_weight()
        self.G = graph
        # self.init_model_weight_RMSprop()

    def init_model_weight(self):
        for i in range(len(self.config.struct) - 1):
            nn.init.xavier_uniform_(self.encoder_list[i].weight)
            nn.init.uniform_(self.encoder_list[i].bias, a=-0.5, b=0.5)
            nn.init.xavier_uniform_(self.decoder_list[i].weight)
            nn.init.uniform_(self.decoder_list[i].bias, a=-0.5, b=0.5)
        nn.init.uniform_(self.query, a=-0.5, b=0.5)

    def init_model_weight_RMSprop(self):
        for i in range(len(self.config.struct) - 1):
            nn.init.normal_(self.encoder_list[i].weight)
            nn.init.uniform_(self.encoder_list[i].bias, a=-0.5, b=0.5)
            nn.init.normal_(self.decoder_list[i].weight)
            nn.init.uniform_(self.decoder_list[i].bias, a=-0.5, b=0.5)
        nn.init.uniform_(self.query, a=-0.5, b=0.5)

    def encoder(self, X):
        for i in range(len(self.config.struct) - 1):
            X = torch.tanh(self.encoder_list[i](X))
        return X

    def decoder(self, X):
        for i in range(len(self.config.struct) - 1):
            X = torch.tanh(self.decoder_list[i](X))
        return X

    def get_context_enc(self, nodes):
        nodes = nodes.cpu().numpy()
        flag = True
        for node in nodes:
            neighbors = list(self.G.g.neighbors(node))
            neighbors_feature = torch.Tensor(self.G.features[neighbors].toarray()).to(device=self.device,
                                                                                      dtype=torch.float32)
            neighbors_enc = self.encoder(neighbors_feature)
            node_feature = torch.Tensor(self.G.features[node].toarray()).to(device=self.device, dtype=torch.float32)
            _node_enc = self.encoder(node_feature)
            node_enc_extend = _node_enc * torch.ones_like(neighbors_enc)
            _cat_node_nei = torch.cat((node_enc_extend, neighbors_enc), dim=1)
            alpha = F.softmax(F.relu(torch.matmul(_cat_node_nei, self.query.t())), dim=0)
            # alpha = F.softmax((node_enc_extend * neighbors_enc).sum(1, keepdim=True), dim=0)
            _context_enc = (alpha * neighbors_enc).sum(0, keepdim=True)
            if flag:
                flag = False
                node_enc = _node_enc
                context_enc = _context_enc
            else:
                node_enc = torch.cat((node_enc, _node_enc), dim=0)
                context_enc = torch.cat((context_enc, _context_enc), dim=0)
        return node_enc, context_enc

    def get_2nd_loss(self, X, new_X):
        B = X * (self.config.beta - 1) + 1
        return torch.pow((new_X - X) * B, 2).sum()

    def get_1st_loss(self, mu_i, sigma_i, mu_j, sigma_j, mu_k, sigma_k):
        e_ij = self._wasserstein_distice(mu_i, sigma_i, mu_j, sigma_j)
        e_ik = self._wasserstein_distice(mu_i, sigma_i, mu_k, sigma_k)
        # return -1.0 * (X * (torch.matmul(encoder, encoder.transpose(0, 1)))).sum()
        # return (torch.pow((s_encoder - t_encoder), 2) * S.view(-1, 1)).sum()
        return (e_ij + torch.exp(-1 * torch.pow(e_ik, 0.5))).sum()

    def _wasserstein_distice(self, mu1, sigma1, mu2, sigma2):
        return (torch.norm(mu1 - mu2, p=2, dim=1) + torch.norm(sigma1 - sigma2, p=2, dim=1))

    def forward(self, data):
        nodes, node_feature, neighbor_mask = data
        # encoder-decoder
        node_enc = self.encoder(node_feature)

        # node_enc, context_enc = self.get_context_enc(nodes)
        # _role_loss = torch.pow(torch.norm(node_enc - context_enc), 2)

        node_dec = self.decoder(node_enc)

        _enc_dec_loss = self.get_2nd_loss(node_feature, node_dec)

        return _enc_dec_loss

    def get_embedding(self, features):
        enc = self.encoder(features)
        return enc

    def save_embedding(self, emb, fout):
        np.save(fout, emb)
        return None


class RDMAE(BasicModule):
    '''
    BiAE: use extend adj and Auto-Encoder
    '''

    def __init__(self, config, graph):
        super(RDMAE, self).__init__()
        self.config = config
        self.model_name = 'Role'
        self.device = torch.device('cuda:{}'.format(config.device) if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(0.5)
        self.encoder_list = nn.ModuleList([
            nn.Linear(self.config.struct[i], self.config.struct[i + 1])
            for i in range(len(self.config.struct) - 1)])
        self.config.struct.reverse()
        self.decoder_list = nn.ModuleList([
            nn.Linear(self.config.struct[i], self.config.struct[i + 1])
            for i in range(len(self.config.struct) - 1)])
        self.config.struct.reverse()
        # self.all_features = features
        # self.nodes_num, self.feature_num = features.size()
        self.query = nn.Parameter(torch.Tensor(1, 2 * self.config.struct[-1]))
        self.init_model_weight()
        self.G = graph
        # self.init_model_weight_RMSprop()

    def init_model_weight(self):
        for i in range(len(self.config.struct) - 1):
            nn.init.xavier_uniform_(self.encoder_list[i].weight)
            nn.init.uniform_(self.encoder_list[i].bias, a=-0.5, b=0.5)
            nn.init.xavier_uniform_(self.decoder_list[i].weight)
            nn.init.uniform_(self.decoder_list[i].bias, a=-0.5, b=0.5)
        nn.init.uniform_(self.query, a=-0.5, b=0.5)

    def init_model_weight_RMSprop(self):
        for i in range(len(self.config.struct) - 1):
            nn.init.normal_(self.encoder_list[i].weight)
            nn.init.uniform_(self.encoder_list[i].bias, a=-0.5, b=0.5)
            nn.init.normal_(self.decoder_list[i].weight)
            nn.init.uniform_(self.decoder_list[i].bias, a=-0.5, b=0.5)
        nn.init.uniform_(self.query, a=-0.5, b=0.5)

    def encoder(self, X):
        for i in range(len(self.config.struct) - 1):
            X = torch.tanh(self.encoder_list[i](X))
        return X

    def decoder(self, X):
        for i in range(len(self.config.struct) - 1):
            X = torch.tanh(self.decoder_list[i](X))
        return X

    def get_context_enc(self, nodes):
        nodes = nodes.cpu().numpy()
        flag = True
        for node in nodes:
            neighbors = list(self.G.g.neighbors(node))
            neighbors_feature = torch.Tensor(self.G.features[neighbors].toarray()).to(device=self.device,
                                                                                      dtype=torch.float32)
            neighbors_enc = self.encoder(neighbors_feature)
            node_feature = torch.Tensor(self.G.features[node].toarray()).to(device=self.device, dtype=torch.float32)
            _node_enc = self.encoder(node_feature)
            node_enc_extend = _node_enc * torch.ones_like(neighbors_enc)
            _cat_node_nei = torch.cat((node_enc_extend, neighbors_enc), dim=1)
            alpha = F.softmax(F.relu(torch.matmul(_cat_node_nei, self.query.t())), dim=0)
            # alpha = F.softmax((node_enc_extend * neighbors_enc).sum(1, keepdim=True), dim=0)
            _context_enc = (alpha * neighbors_enc).sum(0, keepdim=True)
            if flag:
                flag = False
                node_enc = _node_enc
                context_enc = _context_enc
            else:
                node_enc = torch.cat((node_enc, _node_enc), dim=0)
                context_enc = torch.cat((context_enc, _context_enc), dim=0)
        return node_enc, context_enc

    def get_2nd_loss(self, X, new_X):
        B = X * (self.config.beta - 1) + 1
        return torch.abs((new_X - X) * B).sum()

    def get_1st_loss(self, mu_i, sigma_i, mu_j, sigma_j, mu_k, sigma_k):
        e_ij = self._wasserstein_distice(mu_i, sigma_i, mu_j, sigma_j)
        e_ik = self._wasserstein_distice(mu_i, sigma_i, mu_k, sigma_k)
        # return -1.0 * (X * (torch.matmul(encoder, encoder.transpose(0, 1)))).sum()
        # return (torch.pow((s_encoder - t_encoder), 2) * S.view(-1, 1)).sum()
        return (e_ij + torch.exp(-1 * torch.pow(e_ik, 0.5))).sum()

    def _wasserstein_distice(self, mu1, sigma1, mu2, sigma2):
        return (torch.norm(mu1 - mu2, p=2, dim=1) + torch.norm(sigma1 - sigma2, p=2, dim=1))

    def forward(self, data):
        nodes, node_feature, neighbor_mask = data
        # encoder-decoder
        node_enc = self.encoder(node_feature)

        node_enc, context_enc = self.get_context_enc(nodes)
        _role_loss = torch.pow(torch.norm(node_enc - context_enc), 2)

        node_dec = self.decoder(node_enc)

        _enc_dec_loss = self.get_2nd_loss(node_feature, node_dec)
        return _enc_dec_loss + self.config.alpha * _role_loss

    def get_embedding(self, features):
        enc = self.encoder(features)
        return enc

    def save_embedding(self, emb, fout):
        np.save(fout, emb)
        return None


class RDAT(BasicModule):
    '''
    BiAE: use extend adj and Auto-Encoder
    '''

    def __init__(self, config, graph):
        super(RDAT, self).__init__()
        self.config = config
        self.model_name = 'Role'
        self.device = torch.device('cuda:{}'.format(config.device) if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(0.5)
        self.encoder_list = nn.ModuleList([
            nn.Linear(self.config.struct[i], self.config.struct[i + 1])
            for i in range(len(self.config.struct) - 1)])
        self.config.struct.reverse()
        self.decoder_list = nn.ModuleList([
            nn.Linear(self.config.struct[i], self.config.struct[i + 1])
            for i in range(len(self.config.struct) - 1)])
        self.config.struct.reverse()
        # self.all_features = features
        # self.nodes_num, self.feature_num = features.size()
        self.query = nn.Parameter(torch.Tensor(1, 2 * self.config.struct[-1]))
        self.init_model_weight()
        self.G = graph
        # self.init_model_weight_RMSprop()

    def init_model_weight(self):
        for i in range(len(self.config.struct) - 1):
            nn.init.xavier_uniform_(self.encoder_list[i].weight)
            nn.init.uniform_(self.encoder_list[i].bias, a=-0.5, b=0.5)
            nn.init.xavier_uniform_(self.decoder_list[i].weight)
            nn.init.uniform_(self.decoder_list[i].bias, a=-0.5, b=0.5)
        nn.init.uniform_(self.query, a=-0.5, b=0.5)

    def init_model_weight_RMSprop(self):
        for i in range(len(self.config.struct) - 1):
            nn.init.normal_(self.encoder_list[i].weight)
            nn.init.uniform_(self.encoder_list[i].bias, a=-0.5, b=0.5)
            nn.init.normal_(self.decoder_list[i].weight)
            nn.init.uniform_(self.decoder_list[i].bias, a=-0.5, b=0.5)
        nn.init.uniform_(self.query, a=-0.5, b=0.5)

    def encoder(self, X):
        for i in range(len(self.config.struct) - 1):
            X = torch.tanh(self.encoder_list[i](X))
        return X

    def decoder(self, X):
        for i in range(len(self.config.struct) - 1):
            X = torch.tanh(self.decoder_list[i](X))
        return X

    def get_context_enc(self, nodes):
        nodes = nodes.cpu().numpy()
        flag = True
        for node in nodes:
            neighbors = list(self.G.g.neighbors(node))
            neighbors_feature = torch.Tensor(self.G.features[neighbors].toarray()).to(device=self.device,
                                                                                      dtype=torch.float32)
            neighbors_enc = self.encoder(neighbors_feature)
            node_feature = torch.Tensor(self.G.features[node].toarray()).to(device=self.device, dtype=torch.float32)
            _node_enc = self.encoder(node_feature)
            node_enc_extend = _node_enc * torch.ones_like(neighbors_enc)
            _cat_node_nei = torch.cat((node_enc_extend, neighbors_enc), dim=1)
            alpha = F.softmax(F.relu(torch.matmul(_cat_node_nei, self.query.t())), dim=0)
            # alpha = F.softmax((node_enc_extend * neighbors_enc).sum(1, keepdim=True), dim=0)
            _context_enc = (alpha * neighbors_enc).sum(0, keepdim=True)
            if flag:
                flag = False
                node_enc = _node_enc
                context_enc = _context_enc
            else:
                node_enc = torch.cat((node_enc, _node_enc), dim=0)
                context_enc = torch.cat((context_enc, _context_enc), dim=0)
        return node_enc, context_enc

    def get_2nd_loss(self, X, new_X):
        B = X * (self.config.beta - 1) + 1
        return torch.pow((new_X - X) * B, 2).sum()

    def get_1st_loss(self, mu_i, sigma_i, mu_j, sigma_j, mu_k, sigma_k):
        e_ij = self._wasserstein_distice(mu_i, sigma_i, mu_j, sigma_j)
        e_ik = self._wasserstein_distice(mu_i, sigma_i, mu_k, sigma_k)
        # return -1.0 * (X * (torch.matmul(encoder, encoder.transpose(0, 1)))).sum()
        # return (torch.pow((s_encoder - t_encoder), 2) * S.view(-1, 1)).sum()
        return (e_ij + torch.exp(-1 * torch.pow(e_ik, 0.5))).sum()

    def _wasserstein_distice(self, mu1, sigma1, mu2, sigma2):
        return (torch.norm(mu1 - mu2, p=2, dim=1) + torch.norm(sigma1 - sigma2, p=2, dim=1))

    def forward(self, data):
        nodes, node_feature, neighbor_mask = data
        # encoder-decoder
        node_enc = self.encoder(node_feature)

        node_enc, context_enc = self.get_context_enc(nodes)
        _role_loss = torch.pow(torch.norm(node_enc - context_enc), 2)

        node_dec = self.decoder(node_enc)

        # _enc_dec_loss = self.get_2nd_loss(node_feature, node_dec)
        return self.config.alpha * _role_loss

    def get_embedding(self, features):
        enc = self.encoder(features)
        return enc

    def save_embedding(self, emb, fout):
        np.save(fout, emb)
        return None
