
from unittest import TestLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from util import make_tgt_mask
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os


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
            outputs = self.run(x,true_y,pad_id)
            loss = self.model.calculate_loss(outputs, true_y,pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()
            total_acc += self.model.calculate_acc(outputs, true_y,pad_id)
                
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
        model_save_path:Path = Path("best_ponder_models"),
        ):
        with torch.no_grad():
            best_accuracy = 0.0
            best_loss = 100000000
            valid_total_loss = 0.0
            valid_total_acc = 0.0
            self.model.eval()
            for x, true_y in valid_loader:
                x = x.to(self.device)
                true_y = true_y.to(self.device)
                outputs = self.run(x,true_y,pad_id)
                loss = self.model.calculate_loss(outputs, true_y,pad_id)
                valid_total_loss += loss.item()
                valid_total_acc += self.model.calculate_acc(outputs, true_y,pad_id)
                
            if best_accuracy <= valid_total_acc/len(valid_loader.dataset): 
                best_accuracy = valid_total_acc/len(valid_loader.dataset)
                if not os.path.exists(model_save_path) : 
                     os.mkdir(model_save_path)
                torch.save(self.model.state_dict(), model_save_path / Path('state_dict.pt'))
                
            self.writer.add_scalar("Loss/valid", valid_total_loss/len(valid_loader), epoch)
            self.writer.add_scalar("Acc/valid", valid_total_acc/len(valid_loader.dataset), epoch)
            self.writer.flush()
            print(f"valid_loss:{valid_total_loss/len(valid_loader)}")
            print(f"valid_acc:{valid_total_acc/len(valid_loader.dataset)}")

    
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
                outputs = self.run(x,true_y,pad_id)
                test_total_acc += self.model.calculate_acc(outputs, true_y,pad_id=pad_id)
            print(f"test_acc:{test_total_acc/len(test_loader.dataset)}")
            
            
    def analyze(
        self,
        analyze_loader: DataLoader,
        pad_id: int = 0,
    ):
        self.model.eval()
        with torch.no_grad():
            test_total_loss = 0.0
            test_total_acc = 0.0
            self.model.init_analyze()
            for x, true_y in analyze_loader:
                x = x.to(self.device)
                true_y = true_y.to(self.device)
                outputs = self.run(x,true_y,pad_id)
                loss = self.model.calculate_loss(outputs, true_y,pad_id)
                test_total_loss += loss.item()
                test_total_acc += self.model.calculate_acc(outputs, true_y,pad_id=pad_id)
                
                self.model.analyze(x,true_y,outputs,analyze_loader)
                
            self.model.finish_analyze()
            print(f"test_loss:{test_total_loss/len(analyze_loader)}")
            print(f"test_acc:{test_total_acc/len(analyze_loader.dataset)}")