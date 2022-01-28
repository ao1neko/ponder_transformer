from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import string
import more_itertools
from util import make_word_dic,make_id_dic,convert_str
from typing import List,Dict
import random
import itertools
import math


class JsonDataset():
    def __init__(self,json_base_dir:str,json_names:List[str],word_dic=None):
        self.base_dir = Path(json_base_dir)
        self.json_names = [ self.base_dir / json_name for json_name in json_names]      
        self.word_dic = word_dic if word_dic == None else self._make_word_dic()
        self.id_dic = self._make_id_dic(self.word_dic)
        self.vocab_size = len(self.word_dic)
        self.train_data_x = None 
        self.train_data_y = None
        self.valid_data_x = None 
        self.valid_data_y = None
        self.test_data_x = None 
        self.test_data_y = None
        self.make_data(self.word_dic)
    
    def _make_word_dic(self,upper_chr=True,lower_chr=True,min_value=0,max_value=9,operater=["=",",","+","-","*"],tag=["<CLS>","<SEP>"]):
        dic = {"<PAD>":0}
        for c in tag: dic[c] = len(dic)
        if upper_chr == True:
            for c in string.ascii_uppercase: dic[c] = len(dic)
        if lower_chr == True:
            for c in string.ascii_lowercase: dic[c] = len(dic)
        for c in range(min_value,max_value+1): dic[str(c)] = len(dic)
        for c in operater: dic[c] = len(dic)
        return dic
    
    def _make_id_dic(self,word_dic):
        dic = {}
        for key,value in word_dic.items():
            dic[value]=key
        return dic
     
    def make_data(self,word_dic):
        for tail in ["train","valid","test"]:
            data_x = []
            data_y = []
            x_max_len = 0
            y_max_len = 0
            for json_name in self.json_names:
                json_path = self.base_dir / (json_name +"." + tail + "."+"json")
                with open(json_path,'r') as jf:
                    json_dic = json.load(jf)
                for data in json_dic.values():
                    text = [word_dic[c] for c in self._convert_str(data["passage"])]
                    q = [word_dic[c] for c in self._convert_str(data["qa_pairs"][0]["question"])]
                    x = [word_dic["<CLS>"]]+text+[word_dic["<SEP>"]]+q+[word_dic["<SEP>"]]
                    y = [word_dic["<CLS>"]]+[word_dic[c] for c in data["qa_pairs"][0]["answer"]["number"]]+[word_dic["<SEP>"]]
                    x_max_len = max(len(x),x_max_len)               
                    y_max_len = max(len(y),y_max_len)              
                    data_x.append(x)
                    data_y.append(y)     
            for data in data_x: 
                data.extend([0] * (self.x_max_len - len(data)))
            for data in data_y: 
                data.extend([0] * (self.y_max_len - len(data)))                   
            if tail == "train":
                self.train_data_x = torch.tensor(data_x)
                self.train_data_y = torch.tensor(data_y)
            elif tail == "valid":
                self.valid_data_x = torch.tensor(data_x)
                self.valid_data_y = torch.tensor(data_y)
            elif tail == "test":
                self.test_data_x = torch.tensor(data_x)
                self.test_data_y = torch.tensor(data_y)
                
                    
                    
                
class SimpleDataset(Dataset):
    def __init__(self,x,y):
        self.data_x = x
        self.data_y = y
        
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]