# -*- coding: utf-8 -*-
import torch
import time
import torch.nn as nn


class BasicModule(nn.Module):

    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.model')
        torch.save(self.state_dict(), name)
        return name


class Flat(nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)
