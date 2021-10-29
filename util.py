import torch.nn as nn
import torch

def make_tgt_mask(sz):
  mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
  mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
  return mask

def calculate_acc(pred_y,true_y):
  argmax_pred_y = torch.argmax(pred_y, dim=2)
  _,seq_len = true_y.shape
  count = torch.count_nonzero((torch.sum((true_y == argmax_pred_y),1) == seq_len))
  return count.item()