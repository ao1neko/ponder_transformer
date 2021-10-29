import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import json
import string
import tqdm

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
    """

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
        self.id_dic = self._make_id_dic(self.word_dic)
        self.data_x, self.data_y = self._read_json(json_pass,self.word_dic)
        self.x_max_len = max([len(text) for text in self.data_x])
        self.y_max_len = max([len(text) for text in self.data_y])
        
        #padding
        for text in self.data_x:
            text.extend([0] * (self.x_max_len - len(text)))
        for text in self.data_y:
            text.extend([0] * (self.y_max_len - len(text)))
        self.data_x = torch.tensor(self.data_x)
        self.data_y = torch.tensor(self.data_y)
        self.vocab_size = len(self.word_dic)
        
        
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
    
    def _read_json(self,json_pass,word_dic):
        with open(json_pass,'r') as jf:
            json_dic = json.load(jf)
        list_x = []
        list_y = []
        for data in json_dic.values():
            text = [word_dic[c] for c in data["passage"]]
            q = [word_dic[c] for c in data["qa_pairs"][0]["question"]]
            
            list_x.append([word_dic["<CES>"]]+text+[word_dic["<SEP>"]]+q+[word_dic["<SEP>"]])
            list_y.append([word_dic[c] for c in data["qa_pairs"][0]["answer"]["number"]])

        return list_x,list_y
    
    def _make_dic(self,upper_chr=True,lower_chr=True,min_value=0,max_value=9,operater=["=",",","+","-","*"],tag=["<CES>","<SEP>"]):
        dic = {"<PAD>":0}
        if upper_chr == True:
            for c in string.ascii_uppercase: dic[c] = len(dic)
        if lower_chr == True:
            for c in string.ascii_lowercase: dic[c] = len(dic)
        for c in range(min_value,max_value+1): dic[str(c)] = len(dic)
        for c in operater: dic[c] = len(dic)
        for c in tag: dic[c] = len(dic)
        
        return dic
        
    def _make_id_dic(self,word_dic):
        dic = {}
        for key,value in word_dic.items():
            dic[value]=key
        return dic
    
    def id_to_text(self,l):
        return [self.id_dic[id.item()] for id in l]
    
def main():
    m_data = MultiReasoningData("/work01/aoki0903/PonderNet/multihop_experiment/datas/ponder_base.json")
    print(m_data.data_x[0])
    print([m_data.id_dic[int(c)] for c in m_data.data_x[0]])
    
    print(m_data.data_y[0])
    print([m_data.id_dic[int(c)] for c in m_data.data_y[0]])
    

if __name__ == '__main__':
    main()