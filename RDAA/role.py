# -*- coding: utf-8 -*-
import os
import pickle
import random
import time
import warnings

import models
import numpy as np
import pandas as pd
import torch
from config import Roleconfig as config
from data import roleData
from data import roleGraph
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.data import DataLoader
from utils import Task
from utils import utils

random.seed(config.seed)
np.random.seed(config.seed)
torch.cuda.manual_seed(config.seed)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


def train(**kwargs):
    '''
    BiNE use Auto-Encoder, edge batch size
    '''
    config.parse(kwargs)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device(
        'cuda:{}'.format(config.device) if torch.cuda.is_available() else "cpu")
    print(device)
    # if os.path.exists('{}/{}_{}.pkl'.format('cache', config.dataset, 'Graph')):
    #     with open('{}/{}_{}.pkl'.format('cache', config.dataset, 'Graph'), 'rb') as f:
    #         graph = pickle.load(f)
    #     print('exists Graph load it!\nnodes: {}, edges: {}'.format(
    #         graph.g.number_of_nodes(), graph.g.number_of_edges()))
    # else:
    graph = roleGraph(config, sep=' ')
    with open('{}/{}_{}.pkl'.format('cache', config.dataset, 'Graph'), 'wb') as f:
        pickle.dump(graph, f)
    print('success save Graph')
    config.struct[0] = graph.features.shape[1]
    roledata = roleData(graph)
    train_dataloader = DataLoader(roledata, config.batch_size, shuffle=True,
                                  num_workers=config.num_workers)
    graph_adj = graph.get_adjmatrix()
    all_features = torch.Tensor(graph.features.toarray()).to(device=device,
                                                             dtype=torch.float32)

    model = getattr(models, config.model)(config, graph).to(device=device,
                                                            dtype=torch.float32)

    optimizer = torch.optim.Adadelta(model.parameters(), lr=config.lr, rho=0.95,
                                     weight_decay=config.weight_decay)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr,
    # weight_decay=config.weight_decay, alpha=0.9)
    save_path = '../embed/{}/'.format(config.model)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    task = Task(graph, config)
    total_time = 0
    for epoch in range(config.epoch):
        total_loss = 0.0
        start_time = time.time()
        for idx, data in enumerate(train_dataloader):
            nodes = data.numpy()
            node_feature = torch.Tensor(graph.features[nodes].toarray()).squeeze()
            neighbor_adj = torch.Tensor(graph_adj[nodes].toarray())
            data = [data, node_feature, neighbor_adj]
            # neighbors = [list(graph.g.neighbors(node)) for node in nodes]
            # neighbor_feature = torch.Tensor(graph.features[neighbors])
            data = list(map(lambda x: x.to(device=device, dtype=torch.float32), data))
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_time += time.time() - start_time
        print('epoch {0}, loss: {1}, time: {2}'.format(epoch, total_loss,
                                                       time.time() - start_time))
        embedding = model.get_embedding(all_features).data.cpu().numpy()

        if config.model == 'RDAA':
            if (epoch + 1) % 10 == 0:
                task.classfication(utils.cat_neighbor(graph.g, embedding, method='cat_neighbor'),
                                   split_ratio=0.7, loop=20)
            model.save_embedding(embedding, '{}/{}_{}'.format(save_path, config.dataset, epoch))
    embedding = model.get_embedding(all_features).data.cpu().numpy()
    print(total_time)
    ids = np.array(range(embedding.shape[0])).reshape((-1, 1))
    columns = ['id'] + ['x_' + str(i) for i in range(embedding.shape[1])]
    embed = np.concatenate([ids, embedding], axis=1)
    embed = pd.DataFrame(embed, columns=columns)
    embed.to_csv('{}/{}.emb'.format(save_path, config.dataset), index=False)

    for i in np.round(np.linspace(0.1, 0.9, 9), decimals=1):
        task.classfication(utils.cat_neighbor(graph.g, embedding, method='cat_neighbor'),
                           split_ratio=i, loop=100)

def help():
    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | help
    example:
            python {0} train --env='env0701' --lr=0.01
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(config.__class__))
    print(source)


if __name__ == '__main__':
    import fire

    fire.Fire()
