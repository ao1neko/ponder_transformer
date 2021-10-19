import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class Reverse(Dataset):
    def __init__(self,data_size=100,data_dim=40,low = 0,high=10):
        self.data_size = data_size
        self.data_x = np.random.randint(low, high, (data_size, data_dim))
        self.data_y = np.fliplr(self.data_x).copy()
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

