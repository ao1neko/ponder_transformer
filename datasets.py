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
    
    
class MultiReasoningGenBERTData(Dataset):
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
            text = [word_dic[c] for c in self._convert_str(data["passage"])]
            q = [word_dic[c] for c in self._convert_str(data["qa_pairs"][0]["question"])]
            
            list_x.append([word_dic["<CLS>"]]+text+[word_dic["<SEP>"]]+q+[word_dic["<SEP>"]])
            list_y.append([word_dic["<CLS>"]]+[word_dic[c] for c in data["qa_pairs"][0]["answer"]["number"]]+[word_dic["<SEP>"]])

        return list_x,list_y
    
    def _make_dic(self,upper_chr=True,lower_chr=True,min_value=0,max_value=9,operater=["=",",","+","-","*"],tag=["<CLS>","<SEP>"]):
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
    
    def id_to_text(self,l):
        return [self.id_dic[id.item()] for id in l]

    def _convert_str(self,str:str):
        str = str.replace(" ","")
        return str
    
     
class SingleReasoningData(Dataset):
    def __init__(self):
        self.word_dic = self._make_dic()
        self.id_dic = self._make_id_dic(self.word_dic)
        self.data_x, self.data_y = self._make_data(self.word_dic)
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
    
    
    def _make_dic(self,upper_chr=True,lower_chr=True,min_value=0,max_value=9,operater=["=",",","+","-","*"],tag=["<CLS>","<SEP>"]):
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
    
    def _make_data(self,word_dic,min_value:int=0,max_value:int=99):
        list_x = []
        list_y = []
        for a,b in list(itertools.product(range(min_value,max_value+1),range(min_value,max_value+1))):
            answers = []
            answers.append(a+b)
            answers.append(a-b)
            for ope,answer in zip(["+","-"],answers):
                if abs(answer) < 1000:
                    list_x.append([word_dic["<CLS>"]]+[word_dic[c]for c in str(a)]+[word_dic[ope]]+[word_dic[c] for c in str(b)]+[word_dic["<SEP>"]]+[word_dic["="]]+[word_dic["<SEP>"]])
                    list_y.append([word_dic["<CLS>"]]+[word_dic[c] for c in str(answer)]+[word_dic["<SEP>"]])
        all_list = list(zip(list_x,list_y))
        random.shuffle(all_list)
        list_x , list_y= zip(*all_list)
        return list_x,list_y       
        
    def id_to_text(self,l):
        return [self.id_dic[id] for id in l]


    
       
class MultiReasoningData(Dataset):
    def __init__(self,json_pass):
        self.word_dic = self._make_dic()
        self.id_dic = self._make_id_dic(self.word_dic)
        self.data_x, self.data_y = self._read_json(json_pass,self.word_dic)
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
            text = [word_dic[c] for c in self._convert_str(data["passage"])]
            q = [word_dic[c] for c in self._convert_str(data["qa_pairs"][0]["question"])]
            
            list_x.append([word_dic["<CLS>"]]+text+[word_dic["<SEP>"]]+q+[word_dic["<SEP>"]])
            list_y.append(word_dic[data["qa_pairs"][0]["answer"]["number"]])

        return list_x,list_y
    
    def _make_dic(self,upper_chr=True,lower_chr=True,min_value=-1000,max_value=1000,operater=["=",",","+","-","*"],tag=["<CLS>","<SEP>"]):
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
    
    def id_to_text(self,l):
        return [self.id_dic[id.item()] for id in l]

    def _convert_str(self,str:str):
        str = str.replace(","," ,")
        return str.split(" ")
    
    
class ConcatedMultiReasoningData(Dataset):
    def __init__(self,json_pass_list):
        self.word_dic = self._make_dic()
        self.id_dic = self._make_id_dic(self.word_dic)
        self.data_x, self.data_y = self._read_json(json_pass_list,self.word_dic)
        self.x_max_len = max([len(text) for text in self.data_x])
        
        #padding
        for text in self.data_x:
            text.extend([0] * (self.x_max_len - len(text)))
            
        self.data_x = torch.tensor(self.data_x)
        self.data_y = torch.tensor(self.data_y)
        self.vocab_size = len(self.word_dic)
        
        
    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]
    
    def _read_json(self,json_pass_list,word_dic):
        train_list_x = []
        train_list_y = []
        test_list_x = []
        test_list_y = []
        for json_pass in json_pass_list:
            list_x = []
            list_y = []
            with open(json_pass,'r') as jf:
                json_dic = json.load(jf)
            for data in json_dic.values():
                text = [word_dic[c] for c in self._convert_str(data["passage"])]
                q = [word_dic[c] for c in self._convert_str(data["qa_pairs"][0]["question"])]
                
                list_x.append([word_dic["<CLS>"]]+text+[word_dic["<SEP>"]]+q+[word_dic["<SEP>"]])
                list_y.append(word_dic[data["qa_pairs"][0]["answer"]["number"]])
            
            list_x_len = len(list_x)  # n_samples is 60000
            train_len = int(list_x_len * 0.8)
            train_list_x.extend(list_x[:train_len])
            train_list_y.extend(list_y[:train_len])
            test_list_x.extend(list_x[train_len:])
            test_list_y.extend(list_y[train_len:])
        train_list = list(zip(train_list_x,train_list_y))
        test_list = list(zip(test_list_x,test_list_y))
        random.shuffle(train_list)
        random.shuffle(test_list)
        train_list.extend(test_list)
        list_x , list_y= zip(*train_list)
        return list_x,list_y
    
    def _make_dic(self,upper_chr=True,lower_chr=True,min_value=-1000,max_value=1000,operater=["=",",","+","-","*"],tag=["<CLS>","<SEP>"]):
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
    
    def id_to_text(self,l):
        return [self.id_dic[id.item()] for id in l]

    def _convert_str(self,str:str):
        str = str.replace(","," ,")
        return str.split(" ")
    
class SoftReasonersData(Dataset):
    def __init__(self,json_pass):
        self.data_x, self.data_y = self._read_json(json_pass)      
        self.word_dic = None 
        self.id_dic = None 
        self.id_data_x = None
        self.id_data_y = None
        self.id_data_x_len = None
        self.vocab_size = None
        
        
        
    def __len__(self):
        return len(self.id_data_x)

    def __getitem__(self, index):
        return self.id_data_x[index], self.id_data_y[index]
    
    def _read_json(self,json_pass):
        data_x = []
        data_y = []
        
        with open(json_pass,'r') as jsonl_file:
            json_list = list(jsonl_file)
            for json_str in json_list:
                json_dic = json.loads(json_str)
                context = convert_str(json_dic["context"])
                
                for question_dic in json_dic["questions"]:
                    question = convert_str(question_dic["text"])
                    answer = convert_str(str(question_dic["label"]))
                    
                    data_x.append("<CLS> "+ context + " <SEP> " + question + " <SEP>")
                    data_y.append(answer)
        data_x = data_x[:5]
        data_y = data_y[:5]
        
        return data_x,data_y
    
    def def_dic(self,word_dic:Dict[str,int],id_dic:Dict[int,str]):
        self.word_dic = word_dic
        self.vocab_size = len(self.word_dic)
        self.id_dic = {int(k):v for k,v in id_dic.items()}   
            
    def text2id(self,pad_id = 0):
        if self.word_dic is None : 
            print("word_dic is None")
            exit(1)
        id_data_x = []
        id_data_y = []
        
        for text_x,token_y in zip(self.data_x,self.data_y):
            id_list_x = []
            
            for word in text_x.split(" "):
                id_list_x.append(self.word_dic[word])
            id_data_x.append(id_list_x)
            id_data_y.append(self.word_dic[token_y])           
            
        self.id_data_x_len = max([len(text) for text in id_data_x])
        
        #padding
        for text in id_data_x:
            text.extend([pad_id] * (self.id_data_x_len - len(text)))
        
        self.id_data_x = torch.tensor(id_data_x)
        self.id_data_y = torch.tensor(id_data_y)
        
   
class BABIData(Dataset):
    def __init__(self,txt_pass):
        self.data_x, self.data_y = self._read_txt(txt_pass)      
        self.word_dic = None 
        self.id_dic = None 
        self.id_data_x = None
        self.id_data_y = None
        self.id_data_x_len = None
        self.vocab_size = None
        
    def __len__(self):
        return len(self.id_data_x)

    def __getitem__(self, index):
        return self.id_data_x[index], self.id_data_y[index]
    
    def _read_txt(self,txt_pass):
        data_x = []
        data_y = []
        
        with open(txt_pass,'r') as txt_file:
            txt_list = list(txt_file.read().split("\n"))
            txt_list = [txt for txt in txt_list if txt !=''] #最後の\nを削除
            for splited_txt_list in more_itertools.split_before(txt_list,lambda t: t[:2]=="1 "):
                context = []
                for txt in splited_txt_list:
                    if len(txt.split("\t")) == 1:
                        txt = " ".join(txt.split(" ")[1:])
                        txt = convert_str(txt)
                        context.append(txt)
                    else:
                        question, answer, _ = txt.split("\t")
                        question = " ".join(question.split(" ")[1:])
                        
                        question = convert_str(question)
                        answer = convert_str(answer)
                        
                        data_x.append("<CLS> "+ " ".join(context) + " <SEP> " +question + " <SEP>")
                        data_y.append(answer)
        return data_x,data_y
    
    def def_dic(self,word_dic:Dict[str,int],id_dic:Dict[int,str]):
        self.word_dic = word_dic
        self.vocab_size = len(self.word_dic)
        self.id_dic = {int(k):v for k,v in id_dic.items()}   
        
    def text2id(self,pad_id = 0):
        if self.word_dic is None : 
            print("word_dic is None")
            exit(1)
            
        id_data_x = []
        id_data_y = []
        
        for text_x,token_y in zip(self.data_x,self.data_y):
            id_list_x = []
            
            for word in text_x.split(" "):
                id_list_x.append(self.word_dic[word])
            id_data_x.append(id_list_x)
            id_data_y.append(self.word_dic[token_y])           
            
        self.id_data_x_len = max([len(text) for text in id_data_x])
        
        #padding
        for text in id_data_x:
            text.extend([pad_id] * (self.id_data_x_len - len(text)))
        
        self.id_data_x = torch.tensor(id_data_x)
        self.id_data_y = torch.tensor(id_data_y)
    
def main():
    """
    m_data = MultiReasoningData("/work01/aoki0903/PonderNet/multihop_experiment/datas/ponder_base.json")
    print(m_data.data_x[0])
    print([m_data.id_dic[int(c)] for c in m_data.data_x[0]])
    
    print(m_data.data_y[0])
    print([m_data.id_dic[int(c)] for c in m_data.data_y[0]])
    """
    
    t_data = SoftReasonersData("/home/aoki0903/sftp_sync/MyPonderNet/nlp_datasets/rule-reasoning-dataset-V2020.2.5.0/original/depth-3ext/train.jsonl")
    d_data = SoftReasonersData("/home/aoki0903/sftp_sync/MyPonderNet/nlp_datasets/rule-reasoning-dataset-V2020.2.5.0/original/depth-3ext/dev.jsonl")
    s_data = SoftReasonersData("/home/aoki0903/sftp_sync/MyPonderNet/nlp_datasets/rule-reasoning-dataset-V2020.2.5.0/original/depth-3ext/test.jsonl")
    save_pass = "/home/aoki0903/sftp_sync/MyPonderNet/nlp_datasets/rule-reasoning-dataset-V2020.2.5.0/"
    
    make_word_dic([t_data.data_x,d_data.data_x,s_data.data_x],save_pass,init_word_dic={"<PAD>":0,"true":1,"false":2})
    with open(save_pass + "word_dic.json", "r") as tf:
        word_dic = json.load(tf)
        #print(word_dic.items())
        
    make_id_dic(word_dic,save_pass)
    with open(save_pass + "id_dic.json", "r") as tf:
        id_dic = json.load(tf)
        id_dic = {int(k):v for k,v in id_dic.items()}
        print(id_dic.items())
        
    t_data.def_dic(word_dic,id_dic)
    d_data.def_dic(word_dic,id_dic)
    s_data.def_dic(word_dic,id_dic)
    
    t_data.text2id(pad_id=0)
    d_data.text2id(pad_id=0)
    s_data.text2id(pad_id=0)
    
    print([id_dic[word.item()] for word in t_data.id_data_x[0]])
    print([id_dic[word.item()] for word in t_data.id_data_y[0]])
    
    
    t_data = BABIData("/home/aoki0903/sftp_sync/MyPonderNet/nlp_datasets/babi/task_1.txt")
    d_data = BABIData("/home/aoki0903/sftp_sync/MyPonderNet/nlp_datasets/babi/task_2.txt")
    save_pass = "/home/aoki0903/sftp_sync/MyPonderNet/nlp_datasets/babi/"
    
    make_word_dic([t_data.data_x,d_data.data_x],save_pass,init_word_dic={"<PAD>":0})
    
    with open(save_pass + "word_dic.json", "r") as tf:
        word_dic = json.load(tf)
        #print(word_dic.items())
        
    make_id_dic(word_dic,save_pass)
    with open(save_pass + "id_dic.json", "r") as tf:
        id_dic = json.load(tf)
        id_dic = {int(k):v for k,v in id_dic.items()}
        #print(id_dic.items())
    t_data.def_dic(word_dic,id_dic)
    d_data.def_dic(word_dic,id_dic)
    
    t_data.text2id(pad_id=0)
    d_data.text2id(pad_id=0)
    
    print([id_dic[word.item()] for word in t_data.id_data_x[0]])
    print([id_dic[word.item()] for word in t_data.id_data_y[0]])
        
    


if __name__ == '__main__':
    main()