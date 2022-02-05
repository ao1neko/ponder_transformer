from pathlib import Path
import torch
from torch.utils.data import Dataset
import json
import string
from typing import List,Dict


class JsonDataset():
    def __init__(self,json_base_dir:Path,json_names:List[str],word_dic={},id_dic={}):
        self.base_dir = json_base_dir
        self.json_names = [ self.base_dir / Path(json_name) for json_name in json_names]      
        self.word_dic = word_dic 
        self.id_dic = id_dic
        self.vocab_size = len(self.word_dic)
        self.data_x = None 
        self.data_y = None
        self.make_data(self.word_dic)
    
     
    def make_data(self,word_dic):
        data_x = []
        data_y = []
        x_max_len = 0
        y_max_len = 0
        for json_name in self.json_names:
            json_path = self.base_dir / json_name
            with open(json_path,'r') as jf:
                json_dic = json.load(jf)
            for data in json_dic.values():
                text = [word_dic[c] for c in self._convert_str(data["passage"])]
                #q = [word_dic[c] for c in self._convert_str(data["qa_pairs"][0]["question"])]
                #x = [word_dic["<CLS>"]]+text+[word_dic["<SEP>"]]+q+[word_dic["<SEP>"]]
                x = [word_dic["<CLS>"]]+text+[word_dic["<SEP>"]]
                y = [word_dic["<CLS>"]]+[word_dic[c] for c in data["qa_pairs"][0]["answer"]["number"]]+[word_dic["<SEP>"]]
                x_max_len = max(len(x),x_max_len)               
                y_max_len = max(len(y),y_max_len)              
                data_x.append(x)
                data_y.append(y)     
        for data in data_x: 
            data.extend([0] * (x_max_len - len(data)))
        for data in data_y: 
            data.extend([0] * (y_max_len - len(data)))                   
        self.data_x = torch.tensor(data_x)
        self.data_y = torch.tensor(data_y)
            
    def _convert_str(self,str:str):
        str = str.replace(" ","")
        return str             
                    
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]                    

    
    
def make_word_dic(upper_chr=True,lower_chr=True,min_value=0,max_value=9,operater=["=",",","+","-","*"],tag=["<CLS>","<SEP>"]):
    dic = {"<PAD>":0}
    for c in tag: dic[c] = len(dic)
    if upper_chr == True:
        for c in string.ascii_uppercase: dic[c] = len(dic)
    if lower_chr == True:
        for c in string.ascii_lowercase: dic[c] = len(dic)
    for c in range(min_value,max_value+1): dic[str(c)] = len(dic)
    for c in operater: dic[c] = len(dic)
    return dic

def make_id_dic(word_dic):
    dic = {}
    for key,value in word_dic.items():
        dic[value]=key
    return dic