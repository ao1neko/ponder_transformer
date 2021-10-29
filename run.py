import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from ponder_transformer import PonderTransformer,ReconstructionLoss,RegularizationLoss
from vanilla_transformer import Transformer
from util import make_tgt_mask,calculate_acc
from datasets import MultiReasoningData

def main(args):
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    train_data = MultiReasoningData(args.json_pass)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    
    vocab_size = train_data.vocab_size
    max_step=int(args.max_step)
    model = PonderTransformer(vocab_size=vocab_size,allow_halting=False,max_steps=max_step)
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)

    

    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        total_acc = 0.0
        
        model.train()
        for x, true_y in train_loader:
            tgt_mask = make_tgt_mask(true_y.shape[1])
            optimizer.zero_grad() 
            pred_y, p, h = model(x,true_y,tgt_mask=tgt_mask)
            loss_rec_inst = ReconstructionLoss(nn.CrossEntropyLoss())
            loss_reg_inst = RegularizationLoss(lambda_p=1.0/max_step,max_steps=max_step)
            loss_rec = loss_rec_inst(p,pred_y,true_y,)
            loss_reg = loss_reg_inst(p,)
            loss_overall = loss_rec + 0.1 * loss_reg
            loss_overall.backward()
            optimizer.step() 
            
            total_loss += loss_overall.item()
            total_acc += calculate_acc(pred_y[-1],true_y)
        
        print(total_loss/len(train_loader))
        print(total_acc/len(train_data))
        
        
    model.eval()
    x = train_data.data_x
    true_y = train_data.data_y
    pred_y, p, h = model(x,true_y)
    print(calculate_acc(pred_y[-1],true_y)/len(train_data))
    print(train_data.id_to_text(x[0]))
    print(train_data.id_to_text(torch.argmax(pred_y[-1][0],1)))
    print(train_data.id_to_text(true_y[0]))
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明') 
    parser.add_argument('--epochs', default='30') 
    parser.add_argument('--batch_size', default='128') 
    parser.add_argument('--max_step', default='3')
    parser.add_argument('--json_pass', default="/work01/aoki0903/PonderNet/multihop_experiment/datas/ponder_base.json")
    args = parser.parse_args()
    main(args)