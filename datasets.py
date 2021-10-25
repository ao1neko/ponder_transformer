import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import json
import string

class Reverse(Dataset):
    def __init__(self,data_size=100,data_dim=40,low = 0,high=10):
        self.data_size = data_size
        self.data_x = np.random.randint(low, high, (data_size, data_dim))
        self.data_y = np.fliplr(self.data_x).copy()
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]


class ParityDataset(Dataset):
    """Parity of vectors - binary classification dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.

    n_elems : int
        Size of the vectors.

    n_nonzero_min, n_nonzero_max : int or None
        Minimum (inclusive) and maximum (inclusive) number of nonzero
        elements in the feature vector. If not specified then `(1, n_elem)`.
    """z

    def __init__(
        self,
        n_samples,
        n_elems,
        n_nonzero_min=None,
        n_nonzero_max=None,
    ):
        self.n_samples = n_samples
        self.n_elems = n_elems

        self.n_nonzero_min = 1 if n_nonzero_min is None else n_nonzero_min
        self.n_nonzero_max = (
            n_elems if n_nonzero_max is None else n_nonzero_max
        )

        assert 0 <= self.n_nonzero_min <= self.n_nonzero_max <= n_elems

    def __len__(self):
        """Get the number of samples."""
        return self.n_samples

    def __getitem__(self, idx):
        """Get a feature vector and it's parity (target).

        Note that the generating process is random.
        """
        x = torch.zeros((self.n_elems,))
        n_non_zero = torch.randint(
            self.n_nonzero_min, self.n_nonzero_max + 1, (1,)
        ).item()
        x[:n_non_zero] = torch.randint(0, 2, (n_non_zero,)) * 2 - 1
        x = x[torch.randperm(self.n_elems)]

        y = (x == 1.0).sum() % 2

        return x, y
    
    
class MultiReasoningData(Dataset):
    def __init__(self,json_pass):
        self.word_dic = self._make_dic()
        self.data_x, self.data_y = self._read_json(json_pass,self.word_dic)
        
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
    
    def _read_json(json_pass,word_dic):
        with open(json_pass,'r') as jf:
            json_dic = json.load(jf)
        list_x = []
        list_y = []
        for data in json_dic:
            list_x.append([ word_dic[c] for c in data["passage"]])
            list_y.append(data["qa_pairs"]["answer"]["number"])
        return torch.Tensor(list_x),torch.Tensor(list_y)
    
    def _make_dic(upper_chr=True,lower_chr=True,min_value=0,max_value=10000,operater=["=",",","+","-","*"]):
        dic = {"<PAD>":0}
        if upper_chr == True:
            for c in string.ascii_uppercase: dic[c] = len(dic)
        if lower_chr == True:
            for c in string.ascii_lowercase: dic[c] = len(dic)
        for c in range(min_value,max_value+1): dic[str(c)] = len(dic)
        for c in operater: dic[c] = len(dic)
        
        return dic
        
    
    
def main():
    m_data = MultiReasoningData("/work01/aoki0903/PonderNet/multihop_experiment/datas/ponder_base.json")
    print(m_data.data_x[0])
    print(m_data.data_y[0])

if __name__ == '__main__':
    main()