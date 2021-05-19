import torch
import torch.nn as nn
import numpy as np

from Pretrain.Shared_net import SharedNet


class MMDNet(nn.Module):
    def __init__(self):
        super(MMDNet, self).__init__()
        self.fc1 = nn.Linear(in_features=32*32*2, out_features=128)
        self.relu1 = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu1(self.fc1(x))
        return x


class PredictNet(nn.Module):
    def __init__(self):
        super(PredictNet, self).__init__()
        self.fc1 = nn.Linear(in_features=128, out_features=32*32*2)
        self.relu1 = nn.ReLU()
    
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = x.view(x.shape[0], 2, 1, 32, 32)
        return x
