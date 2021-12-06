import torch.nn as nn
import torch
from typing import List, Dict
import json
import math


def make_tgt_mask(sz:int)->torch.Tensor:
  """
  Args:
      sz (int): sequence length

  Returns:
      torch.Tensor: target_mask
  """
  mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
  return mask

    
        
def make_word_dic(data_list:List[List[str]],save_pass,init_word_dic = {"<PAD>":0}):
    word_dic = init_word_dic
    for data in data_list: 
        for text in data: 
            for word in text.split(" "):
                if word not in word_dic:
                    word_dic[word] = len(word_dic) 
    with open(save_pass+ "word_dic.json", "w") as tf:
        json.dump(word_dic,tf)
    
def make_id_dic(word_dic:Dict[str,int],save_pass):
    id_dic = {}
    for key, value in word_dic.items():
        id_dic[value]=key
    with open(save_pass + "id_dic.json", "w") as tf:
        json.dump(id_dic,tf) 
        
def convert_str(str:str):
    str = str.replace("."," .")
    str = str.replace(","," ,")
    str = str.replace("?"," ?")
    str = str.lower()
    return str


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=50):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(1)
            pe = self.pe[:, :seq_len]
            x = x + pe
            return x
