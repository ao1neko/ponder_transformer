import argparse

import torch
import torch.nn as nn
from util import make_tgt_mask, PositionalEncoder,RandomPositionalEncoder
from einops import rearrange, repeat
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from ponder_transformer import ReconstructionLoss, RegularizationLoss, MyRegularizationLoss
from util import make_tgt_mask
import torch.nn.functional as F
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter



class RunModel():
    def __init__(
        self,    
        model: nn.Module,
        device: torch.device,
        writer:SummaryWriter,
    ):
        self.model = model
        self.device = device
        self.writer = writer
        self.model_calculate_loss= self.model.calculate_loss, 
        self.model_calculate_acc = self.model.calculate_acc,
        self.model_init_analyze = self.model.init_analyze,
        self.model_analyze = self.model.analyze,
        self.model_finish_analyze = self.model.finish_analyze
        
    def run(self,x,true_y,pad_id):
        tgt_mask = make_tgt_mask(true_y.shape[1]).to(self.device)
        src_key_padding_mask = (x == pad_id)
        tgt_key_padding_mask = (true_y == pad_id)
        outputs = self.model(x, true_y, tgt_mask=tgt_mask,
                            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return outputs
    
    def train(
        self,
        optimizer: optim,
        train_loader: DataLoader,
        epoch:int,
        pad_id: int = 0,
    ):
        total_loss = 0.0
        total_acc = 0.0
        self.model.train()
        for x, true_y in train_loader:        
            optimizer.zero_grad()
            x = x.to(self.device)
            true_y = true_y.to(self.device)
            outputs = self.run(self,x,true_y,pad_id)
            loss = self.calculate_loss(outputs, true_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
            total_acc += self.calculate_acc(outputs, true_y)
                
            self.writer.add_scalar("Loss/train", total_loss/len(train_loader), epoch)
            self.writer.add_scalar("Acc/train", total_acc/len(train_loader.dataset), epoch)
            self.writer.flush()
            print(f"train_loss:{total_loss/len(train_loader)}")
            print(f"train_acc:{total_acc/len(train_loader.dataset)}")

    
    def valid(
        self,    
        valid_loader: DataLoader = None,
        epoch: int = 0,
        pad_id: int = 0,
        modelsave_path = "best_ponder_models/",
        ):
        
        self.model.eval()
        with torch.no_grad():
            best_accuracy = 0.0
            best_loss = int("inf")
            valid_total_loss = 0.0
            valid_total_acc = 0.0
            for x, true_y in valid_loader:
                x = x.to(self.device)
                true_y = true_y.to(self.device)
                outputs = self.run(self,x,true_y,pad_id)
                loss = self.calculate_loss(outputs, true_y)
                valid_total_loss += loss.item()
                valid_total_acc += self.calculate_acc(outputs, true_y)
                
            if best_accuracy <= valid_total_acc/len(valid_data): 
                best_accuracy = valid_total_acc/len(valid_data)
                torch.save(self.model.state_dict(), modelsave_pass+'_state_dict.pt')
                
            self.writer.add_scalar("Loss/valid", valid_total_loss/len(valid_loader), epoch)
            self.writer.add_scalar("Acc/valid", valid_total_acc/len(valid_data), epoch)
            self.writer.flush()
            print(f"valid_loss:{valid_total_loss/len(valid_loader)}")
            print(f"valid_acc:{valid_total_acc/len(valid_data)}")

    
    def test(
        self,
        test_loader: DataLoader = None,
        pad_id: int = 0,
    ):
        self.model.eval()
        with torch.no_grad():
            test_total_acc = 0.0
            for x, true_y in test_loader:
                x = x.to(self.device)
                true_y = true_y.to(self.device)
                outputs = self.run(self,x,true_y,pad_id)
                test_total_acc += self.calculate_acc(outputs, true_y,pad_id=pad_id)
            print(f"test_acc:{test_total_acc/len(test_data)}")
            
            
    def analyze(
        self,
        test_loader: DataLoader,
        pad_id: int = 0,
    ):
        self.model.eval()
        with torch.no_grad():
            test_total_loss = 0.0
            test_total_acc = 0.0
            self.model_init_analyze()
            for x, true_y in test_loader:
                x = x.to(self.device)
                true_y = true_y.to(self.device)
                outputs = self.run(self,x,true_y,pad_id)
                loss = self.calculate_loss(outputs, true_y)
                test_total_loss += loss.item()
                test_total_acc += self.calculate_acc(outputs, true_y,pad_id=pad_id)
                
                self.model_analyze(x,true_y,outputs,test_loader)
                
            self.model_finish_analyze()
            print(f"test_loss:{test_total_loss/len(test_loader)}")
            print(f"test_acc:{test_total_acc/len(test_data)}")