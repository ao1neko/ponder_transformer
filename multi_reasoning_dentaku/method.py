import numpy as np
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.optim as optim
from distutils.util import strtobool
import random
from torch.utils.data.dataset import Subset
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim





def train(
  model: nn.Module,
  optimizer: optim,
  train_dataset,
  ):
    output = model(**small_train_dataset)
    output.loss.backward()
    optimizer.step()  
    
    
    total_loss = 0.0
    total_acc = 0.0
    model.train()
    for x, true_y in train:        
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
        
         
def valid():
    output = model.generate(**small_eval_dataset)
    #print(output)
    print([tokenizer.decode(g, skip_special_tokens=False, clean_up_tokenization_spaces=False) for g in output[0]])

