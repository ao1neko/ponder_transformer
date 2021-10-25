import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ponder_transformer import PonderTransformer
from vanilla_transformer import Transformer
from util import make_tgt_mask
from datasets import MultiReasoningData

def main(args):
    vocab_size = 500
    model = PonderTransformer(vocab_size=vocab_size,allow_halting=True)
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    
    
    train_data = MultiReasoningData(args.json_pass)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    sub_x = torch.randint(low=0,high=vocab_size-1,size=(2,50))
    sub_true_y = torch.randint(low=0,high=vocab_size-1,size=(2,5))
    print(sub_x.shape)
    print(sub_true_y.shape)
    
    for epoch in range(epochs):
        model.train()
        for x, true_y in train_loader:
            print(x.shape)
            print(true_y.shape)
            tgt_mask = make_tgt_mask(true_y.shape[1])
            optimizer.zero_grad() 
            exit()
            pred_y, p, h = model(x,true_y,tgt_mask=tgt_mask)
            print(pred_y.shape)
            #loss = loss_fn(predict_y, y)
            #loss.backward()
            optimizer.step() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明') 
    parser.add_argument('--epochs', default='100') 
    parser.add_argument('--batch_size', default='128') 
    parser.add_argument('--json_pass', default="/work01/aoki0903/PonderNet/multihop_experiment/datas/ponder_base.json")
    args = parser.parse_args()
    main(args)