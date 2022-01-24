import numpy as np
import math
import os
import gc
import sys 
sys.path.append("./utils")
sys.path.append("./models")
import time
from glob import glob
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd as autograd 
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from multiprocessing import Manager

from process_data import FeatureIndex
from utils.parse_utils import *   



# LOAD DATA
class CMDDataset(Dataset):
    def __init__(self, 
                 x=None,
                 k=None,
                 n=None,
                 m=None,
                 y=None,
                 device=None):
        manager = Manager()
        self.x = manager.list(x)
        self.k = manager.list(k)
        self.n = manager.list(n)
        self.m = manager.list(m)
        self.y = manager.list(y)
        self.device = device

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
       
        x = np.load(self.x[idx]).astype(np.float32)
        k = np.load(self.k[idx]).astype(np.float32)
        n = np.load(self.n[idx]).astype(np.float32)
        m = np.load(self.m[idx]).astype(np.float32)
        y = np.load(self.y[idx]).astype(np.float32)

        x_ = np.argmax(x, axis=-1)
        y_ = np.argmax(y, axis=-1) 
        k_ = np.asarray(np.argmax(k[0,:12], axis=-1))

        # c label
        clab_ = np.asarray(len(np.unique(y_))) # number of chords

        return x_, k_, n, m, y_, clab_

class CMDPadCollate(object):
    '''
    Ref: https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/7
    '''

    def __init__(self, dim=0):
        self.dim = dim

    def pad_matrix(self, inp, pad, pad_dim=None):
        if pad_dim is 0:
            padded = torch.zeros([pad, inp.size(1)])
            padded[:inp.size(0), :inp.size(1)] = inp 
            padded = padded.transpose(0, 1)
        elif pad_dim is 1:
            padded = torch.zeros([inp.size(0), pad])
            padded[:inp.size(0), :inp.size(1)] = inp 

        return padded

    def pad_collate(self, batch):

        xs = [torch.from_numpy(b[0]) for b in batch]
        ks = [torch.from_numpy(b[1]) for b in batch]
        ns = [torch.from_numpy(b[2]) for b in batch]
        ms = [torch.from_numpy(b[3]) for b in batch]
        ys = [torch.from_numpy(b[4]) for b in batch]
        cs = [torch.from_numpy(b[5]) for b in batch]

        # stack all
        xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=88)
        ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=72)
        ns = map(lambda x: 
            self.pad_matrix(x, pad=xs.size(1), pad_dim=0), ns)
        ns = nn.utils.rnn.pad_sequence(list(ns), batch_first=True).transpose(1, 2)
        ms = map(lambda x: 
            self.pad_matrix(x, pad=ys.size(1), pad_dim=1), ms)
        ms = nn.utils.rnn.pad_sequence(list(ms), batch_first=True)
        ks = torch.stack(list(ks), dim=0)
        cs = torch.stack(list(cs), dim=0)

        return xs, ks, ns, ms, ys, cs

    def __call__(self, batch):
        return self.pad_collate(batch)

class HLSDDataset(Dataset):
    def __init__(self, 
                 x=None,
                 k=None,
                 n=None,
                 m=None,
                 y=None,
                 device=None):
        manager = Manager()
        self.x = manager.list(x)
        self.k = manager.list(k)
        self.n = manager.list(n)
        self.m = manager.list(m)
        self.y = manager.list(y)
        self.device = device

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
       
        x = np.load(self.x[idx]).astype(np.float32)
        k = np.load(self.k[idx]).astype(np.float32)
        n = np.load(self.n[idx]).astype(np.float32)
        m = np.load(self.m[idx]).astype(np.float32)
        y = np.load(self.y[idx]).astype(np.float32)

        x_ = np.argmax(x, axis=-1)
        y_ = np.argmax(y, axis=-1) 
        k_ = np.asarray(12 * k)

        # c label
        clab_ = np.asarray(len(np.unique(y_))) # number of chords
        
        return x_, k_, n, m, y_, clab_

class HLSDPadCollate(object):
    '''
    Ref: https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/7
    '''

    def __init__(self, dim=0):
        self.dim = dim

    def pad_matrix(self, inp, pad, pad_dim=None):
        if pad_dim is 0:
            padded = torch.zeros([pad, inp.size(1)])
            padded[:inp.size(0), :inp.size(1)] = inp 
            padded = padded.transpose(0, 1)
        elif pad_dim is 1:
            padded = torch.zeros([inp.size(0), pad])
            padded[:inp.size(0), :inp.size(1)] = inp 

        return padded

    def pad_collate(self, batch):

        xs = [torch.from_numpy(b[0]) for b in batch]
        ks = [torch.from_numpy(b[1]) for b in batch]
        ns = [torch.from_numpy(b[2]) for b in batch]
        ms = [torch.from_numpy(b[3]) for b in batch]
        ys = [torch.from_numpy(b[4]) for b in batch]
        cs = [torch.from_numpy(b[5]) for b in batch]

        # stack all
        xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=88)
        ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=72)
        ns = map(lambda x: 
            self.pad_matrix(x, pad=xs.size(1), pad_dim=0), ns)
        ns = nn.utils.rnn.pad_sequence(list(ns), batch_first=True).transpose(1, 2)
        ms = map(lambda x: 
            self.pad_matrix(x, pad=ys.size(1), pad_dim=1), ms)
        ms = nn.utils.rnn.pad_sequence(list(ms), batch_first=True)
        ks = torch.stack(list(ks), dim=0)
        cs = torch.stack(list(cs), dim=0)

        return xs, ks, ns, ms, ys, cs

    def __call__(self, batch):
        return self.pad_collate(batch)


