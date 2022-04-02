
from turtle import mode
from unittest import TestLoader
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
from tqdm import tqdm

from torch.utils.data import DataLoader,TensorDataset
from util import make_tgt_mask
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os


class RuBaseModel():
    def __init__(
        self,    
        model: nn.Module,
        device: torch.device,
        writer:SummaryWriter,
        word_dic:Dict,
        id_dic:Dict,
    ):
        self.model = model
        self.device = device
        self.writer = writer
        self.word_dic=word_dic
        self.id_dic = id_dic           
         
    def train(
        self,
        optimizer: optim,
        epochs:int,
        model_save_path:Path = Path("best_ponder_models"),
        pad_id: int = 0,
    ):
        for epoch in tqdm(range(epochs)):
            total_loss = 0.0
            total_acc = 0.0
            self.model.train()
            for x, true_y in self.train_data:        
                optimizer.zero_grad()
                x = x.to(self.device)
                true_y = true_y.to(self.device)
                
                #print([self.id_dic[xx.item()] for xx in x[0]])
                #print([self.id_dic[xx.item()] for xx in true_y[0]])
                #exit()
                
                outputs = self.run(x,true_y,pad_id)
                loss = self.model.calculate_loss(outputs, true_y,pad_id)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
                total_loss += loss.item()
                total_acc += self.model.calculate_acc(outputs, true_y,pad_id)
                    
            self.writer.add_scalar("Loss/train", total_loss/len(self.train_data), epoch)
            self.writer.add_scalar("Acc/train", total_acc/len(self.train_data.dataset), epoch)
            self.writer.flush()
            print(f"train_loss:{total_loss/len(self.train_data)}")
            print(f"train_acc:{total_acc/len(self.train_data.dataset)}")
            self.valid(
                epoch=epoch,
                pad_id=pad_id,
                model_save_path=model_save_path
                )

    def run(self,x,true_y,pad_id):
        tgt_mask = make_tgt_mask(true_y.shape[1]).to(self.device)
        src_key_padding_mask = (x == pad_id)
        tgt_key_padding_mask = (true_y == pad_id)
        outputs = self.model(x, true_y, tgt_mask=tgt_mask,
                            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return outputs
            
    def valid(
        self,    
        epoch: int = 0,
        pad_id: int = 0,
        model_save_path:Path = Path("best_ponder_models"),
        ):
        with torch.no_grad():
            best_loss = 100000000
            valid_total_loss = 0.0
            valid_total_acc = 0.0
            self.model.eval()
            for x, true_y in self.valid_data:
                x = x.to(self.device)
                true_y = true_y.to(self.device)
                outputs = self.run(x,true_y,pad_id)
                loss = self.model.calculate_loss(outputs, true_y,pad_id)
                valid_total_loss += loss.item()
                valid_total_acc += self.model.calculate_acc(outputs, true_y,pad_id)
                
            if valid_total_loss/len(self.valid_data) < best_loss: 
                best_loss = valid_total_loss/len(self.valid_data)
                if not os.path.exists(model_save_path) : 
                    os.makedirs(model_save_path)
                torch.save(self.model.state_dict(), model_save_path / Path('state_dict.pt'))
                
            self.writer.add_scalar("Loss/valid", valid_total_loss/len(self.valid_data), epoch)
            self.writer.add_scalar("Acc/valid", valid_total_acc/len(self.valid_data.dataset), epoch)
            self.writer.flush()
            print(f"valid_loss:{valid_total_loss/len(self.valid_data)}")
            print(f"valid_acc:{valid_total_acc/len(self.valid_data.dataset)}")

    
    def test(
        self,
        pad_id: int = 0,
    ):
        self.model.eval()
        with torch.no_grad():
            test_total_loss = 0.0
            test_total_acc = 0.0
            for x, true_y in self.test_data:
                x = x.to(self.device)
                true_y = true_y.to(self.device)
                outputs = self.run(x,true_y,pad_id)
                loss = self.model.calculate_loss(outputs, true_y,pad_id)
                test_total_loss += loss.item()
                test_total_acc += self.model.calculate_acc(outputs, true_y,pad_id=pad_id)
                
                
            print(f"test_loss:{test_total_loss/len(self.test_data)}")
            print(f"test_acc:{test_total_acc/len(self.test_data.dataset)}")
            
            
    def analyze(
        self,
        pad_id: int = 0,
    ):
        self.model.eval()
        with torch.no_grad():
            test_total_loss = 0.0
            test_total_acc = 0.0
            self.model.init_analyze()
            for x, true_y in self.analyze_data:
                x = x.to(self.device)
                true_y = true_y.to(self.device)
                outputs = self.run(x,true_y,pad_id)
                loss = self.model.calculate_loss(outputs, true_y,pad_id)
                test_total_loss += loss.item()
                test_total_acc += self.model.calculate_acc(outputs, true_y,pad_id=pad_id)
                
                self.model.analyze(x,true_y,outputs,self.id_dic)
                
            self.model.finish_analyze()
            print(f"test_loss:{test_total_loss/len(self.analyze_data)}")
            print(f"test_acc:{test_total_acc/len(self.analyze_data.dataset)}")
    

    def _list_to_tensor(self,list_dataset:List[str],batch_size):
        pass

    def _test_list_to_tensor(self,list_dataset:List[str],batch_size):
        pass
    






class RunVanillaModel(RuBaseModel):    
    def __init__(
        self,    
        model: nn.Module,
        device: torch.device,
        writer:SummaryWriter,
        word_dic:Dict,
        id_dic:Dict,
        batch_size:int,
        train_data:List[str],
        valid_data:List[str],
        test_data:List[str],
    ):
        super().__init__(model=model,device=device,writer=writer,word_dic=word_dic,id_dic=id_dic,)
        self.train_data =self._list_to_tensor(train_data.data,batch_size) 
        self.valid_data =self._list_to_tensor(valid_data.data,batch_size)
        self.test_data =self._test_list_to_tensor(test_data.data,1)
        self.analyze_data = self.test_data      
        
        
    def _list_to_tensor(self,list_dataset:List[str],batch_size):
        data_x=[]
        data_y=[]        
        x_max_len = 0
        y_max_len = 0 
        for data_list in list_dataset:
            #data_list:{x1,x2..}
            for i in range(len(data_list)-1):
                data_x.append([self.word_dic["<CLS>"]]+data_list[i]+[self.word_dic["<SEP>"]])
                data_y.append([self.word_dic["<CLS>"]]+data_list[i+1]+[self.word_dic["<SEP>"]])
            x_max_len = max(max([len(x)+2 for x in data_list][:-1]),x_max_len)
            y_max_len = max(max([len(x)+2 for x in data_list][1:]),y_max_len)
        for data in data_x: 
            data.extend([0] * (x_max_len - len(data)))     
        for data in data_y: 
            data.extend([0] * (y_max_len - len(data)))  
            
        tensor_dataset = TensorDataset(torch.tensor(data_x), torch.tensor(data_y))
        train_loader = torch.utils.data.DataLoader(dataset=tensor_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
        return train_loader

    def _test_list_to_tensor(self,list_dataset:List[str],batch_size):
        data_x=[]
        data_y=[]        
        max_len = 0
        for data_list in list_dataset:
            max_len = max(max([len(x)+2 for x in data_list]),max_len)
        for data_list in list_dataset: 
            for i in range(len(data_list)):
                data_list[i] = [self.word_dic["<CLS>"]]+data_list[i]+[self.word_dic["<SEP>"]]
                data_list[i].extend([0] * (max_len - len(data_list[i])))
                
                
        for data_list in list_dataset:
            #data_list:{x1,x2..}
            data_x.append(data_list[0])
            data_y.append(data_list[1:])       
        tensor_dataset = TensorDataset(torch.tensor(data_x), torch.tensor(data_y))
        train_loader = torch.utils.data.DataLoader(dataset=tensor_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)
        return train_loader
    
    def test(
        self,
        pad_id: int = 0,
    ):
        self.model.eval()
        with torch.no_grad():
            test_total_acc = 0.0
            for x, true_y in self.test_data:
                input = x
                true_y = true_y.to(self.device)
                _ , loop_size, seq_size = true_y.shape
                acc_flag = True
                
                for i in range(loop_size):
                    input = input.to(self.device)
                    true_y_step = true_y[0][i].unsqueeze(0)
                    
                    outputs = self.run(input,true_y_step,pad_id)
                    input = true_y_step
                    if not self.model.calculate_acc(outputs, true_y_step,pad_id=pad_id):
                        acc_flag = False
                        break
                    
                if acc_flag == True:
                    test_total_acc += 1
            print(f"test_acc:{test_total_acc/len(self.test_data.dataset)}")
            
    def analyze(
        self,
        pad_id: int = 0,
    ):
        self.model.eval()
        with torch.no_grad():
            test_total_acc = 0.0
            self.model.init_analyze()
            for x, true_y in self.analyze_data:
                input = x.to(self.device)
                true_y = true_y.to(self.device)
                _ , loop_size, seq_size = true_y.shape
                acc_flag = True
                for i in range(loop_size):
                    input = input.to(self.device)
                    true_y_step = true_y[0][i].unsqueeze(0)
                    
                    outputs = self.run(input,true_y_step,pad_id)
                    self.model.analyze(input,true_y_step,outputs,self.id_dic)
                    input = true_y_step
                    
                    if not self.model.calculate_acc(outputs, true_y_step,pad_id=pad_id):
                        acc_flag = False
                        break

                if acc_flag == True:
                    test_total_acc += 1
                
            self.model.finish_analyze()
            print(f"test_acc:{test_total_acc/len(self.analyze_data.dataset)}")
            
                                      
class RunDecoderModel(RuBaseModel):    
    def __init__(
        self,    
        model: nn.Module,
        device: torch.device,
        writer:SummaryWriter,
        word_dic:Dict,
        id_dic:Dict,
        batch_size:int,
        train_data:List[str],
        valid_data:List[str],
        test_data:List[str],
    ):
        super().__init__(model=model,device=device,writer=writer,word_dic=word_dic,id_dic=id_dic,)
        self.train_data =self._list_to_tensor(train_data.data,batch_size) 
        self.valid_data =self._list_to_tensor(valid_data.data,batch_size)
        self.test_data =self._list_to_tensor(test_data.data,1)
        self.analyze_data = self.test_data  
        
            
    def _list_to_tensor(self,list_dataset:List[str],batch_size):
        data_x=[]
        data_y=[]        
        x_max_len = 0
        y_max_len = 0 
        for data_list in list_dataset:
            #data_list:{x1,x2..}
            data_x.append([self.word_dic["<CLS>"]]+data_list[0]+[self.word_dic["<SEP>"]])
            data_y.append([self.word_dic["<CLS>"]]+self._split_list(data_list[1:],self.word_dic["="])+[self.word_dic["<SEP>"]])
            x_max_len = max(len(data_x[-1]),x_max_len)
            y_max_len = max(len(data_y[-1]),y_max_len)
        for data in data_x: 
            data.extend([0] * (x_max_len - len(data)))     
        for data in data_y: 
            data.extend([0] * (y_max_len - len(data)))  
            
        tensor_dataset = TensorDataset(torch.tensor(data_x), torch.tensor(data_y))
        train_loader = torch.utils.data.DataLoader(dataset=tensor_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
        return train_loader

    def _split_list(self,list_list:List[List[str]],s:str=None):
        temp_list = list_list[0]
        for i in range(len(list_list)-1): 
            if s is not None: temp_list+= [s]
            temp_list+=list_list[i+1]
        return temp_list
    
            
            
                                     
class RunDecoderMultiStepModel(RuBaseModel):    
    def __init__(
        self,    
        model: nn.Module,
        device: torch.device,
        writer:SummaryWriter,
        word_dic:Dict,
        id_dic:Dict,
        batch_size:int,
        train_data:List[str],
        valid_data:List[str],
        test_data:List[str],
    ):
        super().__init__(model=model,device=device,writer=writer,word_dic=word_dic,id_dic=id_dic,)
        self.train_data =self._list_to_tensor(train_data.data,batch_size) 
        self.valid_data =self._list_to_tensor(valid_data.data,batch_size)
        self.test_data =self._test_list_to_tensor(test_data.data,1)
        self.analyze_data = self.test_data  

    def _list_to_tensor(self,list_dataset:List[str],batch_size):
        data_x=[]
        data_y=[]        
        x_max_len = 0
        y_max_len = 0 
        for data_list in list_dataset:
            #data_list:{x1,x2..}
            for i in range(len(data_list)-1):
                data_x.append([self.word_dic["<CLS>"]]+data_list[i]+[self.word_dic["<SEP>"]])
                data_y.append([self.word_dic["<CLS>"]]+data_list[i+1]+[self.word_dic["<SEP>"]])
            x_max_len = max(max([len(x)+2 for x in data_list][:-1]),x_max_len)
            y_max_len = max(max([len(x)+2 for x in data_list][1:]),y_max_len)
        for data in data_x: 
            data.extend([0] * (x_max_len - len(data)))     
        for data in data_y: 
            data.extend([0] * (y_max_len - len(data)))  
            
        tensor_dataset = TensorDataset(torch.tensor(data_x), torch.tensor(data_y))
        train_loader = torch.utils.data.DataLoader(dataset=tensor_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
        return train_loader

    def _test_list_to_tensor(self,list_dataset:List[str],batch_size):
        data_x=[]
        data_y=[]        
        max_len = 0
        for data_list in list_dataset:
            max_len = max(max([len(x)+2 for x in data_list]),max_len)
        for data_list in list_dataset: 
            for i in range(len(data_list)):
                data_list[i] = [self.word_dic["<CLS>"]]+data_list[i]+[self.word_dic["<SEP>"]]
                data_list[i].extend([0] * (max_len - len(data_list[i])))
                
                
        for data_list in list_dataset:
            #data_list:{x1,x2..}
            data_x.append(data_list[0])
            data_y.append(data_list[1:])       
        tensor_dataset = TensorDataset(torch.tensor(data_x), torch.tensor(data_y))
        train_loader = torch.utils.data.DataLoader(dataset=tensor_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False)
        return train_loader        
            

    
    def test(
        self,
        pad_id: int = 0,
    ):
        self.model.eval()
        with torch.no_grad():
            test_total_acc = 0.0
            for x, true_y in self.test_data:
                input = x
                true_y = true_y.to(self.device)
                _ , loop_size, seq_size = true_y.shape
                acc_flag = True
                
                for i in range(loop_size):
                    input = input.to(self.device)
                    true_y_step = true_y[0][i].unsqueeze(0)
                    
                    outputs = self.run(input,true_y_step,pad_id)
                    input = true_y_step
                    if not self.model.calculate_acc(outputs, true_y_step,pad_id=pad_id):
                        acc_flag = False
                        break
                    
                if acc_flag == True:
                    test_total_acc += 1
            print(f"test_acc:{test_total_acc/len(self.test_data.dataset)}")
            
    def analyze(
        self,
        pad_id: int = 0,
    ):
        self.model.eval()
        with torch.no_grad():
            test_total_acc = 0.0
            self.model.init_analyze()
            for x, true_y in self.analyze_data:
                input = x.to(self.device)
                true_y = true_y.to(self.device)
                _ , loop_size, seq_size = true_y.shape
                acc_flag = True
                for i in range(loop_size):
                    input = input.to(self.device)
                    true_y_step = true_y[0][i].unsqueeze(0)
                    
                    outputs = self.run(input,true_y_step,pad_id)
                    self.model.analyze(input,true_y_step,outputs,self.id_dic)
                    input = true_y_step
                    
                    if not self.model.calculate_acc(outputs, true_y_step,pad_id=pad_id):
                        acc_flag = False
                        break

                if acc_flag == True:
                    test_total_acc += 1
                
            self.model.finish_analyze()
            print(f"test_acc:{test_total_acc/len(self.analyze_data.dataset)}")
            
    
            
            
