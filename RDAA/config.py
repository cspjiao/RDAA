# -*- coding: utf-8 -*-
import warnings


class BasicConfig():
    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

    def show(self):
        print('user config:')
        for k, v in self.__class__.__dict__.iteritems():
            if not k.startswith('__'):
                print(k, getattr(self, k))

    def save(self, file_out=None):
        for k, v in self.__class__.__dict__.iteritems():
            if not k.startswith('__'):
                if file_out:
                    print >> file_out, (k, getattr(self, k))


class RoleConfig(BasicConfig):
    dataset = 'usa-flights'
    directed = False
    weighted = False
    weight_decay = 1e-2
    batch_size = 32
    lr = 0.01
    epoch = 100
    num_workers = 4
    split_ratio = 0.8
    snapshoot = True
    save_dataset = False
    beta = 10
    alpha = 5
    gamma = 1.0
    struct = [-1, 128, 128]
    seed = 678
    k = 0
    snapshoot = False
    device = 0
    out = None
    speed = False
    model = 'RDAA'


Roleconfig = RoleConfig()
