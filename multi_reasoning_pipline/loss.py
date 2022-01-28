import argparse

import torch
import torch.nn as nn
from util import make_tgt_mask, PositionalEncoder,RandomPositionalEncoder
from einops import rearrange, repeat
import math
import matplotlib.pyplot as plt

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    def forward(self, p, y_pred, y_true, pad_id=0):
        max_steps, batch_size, seq_len, _ = y_pred.shape
        y_pred = rearrange(y_pred, 'n b s d-> n d b s')
        y_pred_sequence= y_pred[:,:,:,:-1].clone()
        y_true_sequence = y_true[:,1:].clone()
        y_true_sequence = y_true_sequence.expand(max_steps,batch_size,seq_len-1)
        mask = (y_true_sequence != pad_id).float()   
        return (p * (self.loss_func(y_pred_sequence,y_true_sequence)*mask).sum(dim=-1)/torch.count_nonzero(mask,dim=-1)).sum(dim=0).mean()


class RegularizationLoss(nn.Module):
    def __init__(self, lambda_p, device=torch.device('cpu'), max_steps=20):
        super().__init__()

        p_g = torch.zeros((max_steps,)).to(device)
        not_halted = 1.0

        for k in range(max_steps):
            p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)

        self.register_buffer("p_g", p_g)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, p):
        steps, batch_size = p.shape
        p = p.transpose(0, 1)  # (batch_size, max_steps)
        p_g_batch = self.p_g[None, :steps].expand_as(
            p
        )  # (batch_size, max_steps)
        return self.kl_div(p, p_g_batch)
    
    
    
class SequentialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction='none')

    def forward(self,y_pred, y_true, pad_id=0):
        y_true_sequence = y_true[:,1:].clone()
        y_pred_sequence  = y_pred[:,:-1].clone()
        y_pred_sequence = rearrange(y_pred_sequence, 'b s d-> b d s')
        mask = (y_true_sequence != pad_id).float()
        
        return (((self.loss_func(y_pred_sequence,y_true_sequence)*mask).sum(dim=-1))/torch.count_nonzero(mask,dim=-1)).mean()
        #return (self.loss_func(y_pred_sequence,y_true_sequence)*mask).sum(dim=-1).mean()

