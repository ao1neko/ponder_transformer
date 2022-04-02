from pathlib import Path
import json
from typing import List,Dict
import string


class OnestepDataset():
    def __init__(self,json_base_dir:Path,json_names:List[str]):
        self.base_dir = json_base_dir
        self.json_names = [ self.base_dir / Path(json_name) for json_name in json_names]      
        self.word_dic = make_word_dic() 
        self.id_dic = make_id_dic(self.word_dic)
        self.vocab_size = len(self.word_dic)
        self.data = [] 
        self.make_data(self.word_dic)
    
     
    def make_data(self,word_dic):
        for json_name in self.json_names:
            json_path = self.base_dir / json_name
            with open(json_path,'r') as jf:
                json_dic = json.load(jf)
            for data in json_dic.values():
                id_datas = [[word_dic[c] for c in self._convert_str(x)] for x in data["datas"]]
                self.data.append(id_datas)

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