import os
import os.path as osp
import numpy as np

import torch
import torchvision
from torch.utils import data

import glob
from sklearn.preprocessing import normalize
import random

class NFormerDataset(data.Dataset):
    def __init__(self, data, data_length = 7000):
        self.data_length = data_length
        self.feats = data[0]
        self.ids = data[1]
        self.data_num = self.feats.shape[0]

    def __len__(self):
        return self.feats.shape[0]//30

    def __getitem__(self, index):
        center_index = random.randint(0, self.data_num - 1)
        center_feat = self.feats[center_index].unsqueeze(0)
        center_pid = self.ids[center_index]
    
        selected_flags = torch.zeros(self.data_num)
        selected_flags[center_index] = 1
        distmat = 1 - torch.mm(center_feat, self.feats.transpose(0,1))
        indices = torch.argsort(distmat, dim=1).numpy()
        indices = indices[0,:int(self.data_length * (1 + random.random()))].tolist()
        indices = random.sample(indices,self.data_length)
        
        random.shuffle(indices)
        feat_ = self.feats[indices]
        id_ = self.ids[indices]
        

        return feat_, id_


