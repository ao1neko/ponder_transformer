import argparse
import numpy as np
import torch
import torch.optim as optim
from distutils.util import strtobool
import random
import datetime
from torch.utils.data.dataset import Subset
import sys
import os
sys.path.append(os.pardir)

from ponder_transformer import PonderTransformer
from vanilla_transformer import Transformer
from loop_transformer import LoopTransformerGenerater
from datasets import MultiReasoningData,ConcatedMultiReasoningData,SingleReasoningData


import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from run_ponder_transformer import ponder_train, ponder_test, ponder_print_sample
from run_vanilla_transformer import vanilla_train, vanilla_test, vanilla_print_sample


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    args_str = '_'+','.join([arg+"="+str(getattr(args, arg))
                            for arg in vars(args) if arg not in ['json_pass', 'load_pass', 'log_dir']])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    batch_size = args.batch_size
    beta = args.beta
    ponder_model = strtobool(args.ponder_model)
    if ponder_model:
        log_dir = 'runs_shuffle/ponder/pretrain/' + args_str
        save_pass = 'best_ponder_models_shuffle/pretrain'
    else:
        log_dir = 'runs_shuffle/vanilla/pretrain/'  + args_str
        save_pass = 'best_vanilla_models_shuffle/pretrain'
        
    writer = SummaryWriter(log_dir=log_dir, comment=log_dir)
    
    train_data = SingleReasoningData() 
    valid_data = SingleReasoningData() 
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    vocab_size = train_data.vocab_size
    max_step = args.max_step
    lambda_p=args.lambda_p
    emb_dim = args.emb_dim
    liner_dim = args.liner_dim
    load_pass = args.load_pass

    if ponder_model:
        model = PonderTransformer(vocab_size=vocab_size, allow_halting=False,emb_dim=emb_dim,
                                  max_steps=max_step, num_token=vocab_size,liner_dim=liner_dim).to(device)
    else:
        model = Transformer(vocab_size=vocab_size,emb_dim=emb_dim,
                            num_token=vocab_size, num_layers=max_step,liner_dim=liner_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,)

    sep_id = train_data.word_dic["<SEP>"]
    pad_id = train_data.word_dic["<PAD>"]

    if ponder_model:
        if strtobool(args.train):
            ponder_train(
                model=model,
                optimizer=optimizer,
                train_data=train_data,
                train_loader=train_loader,
                max_step=max_step,
                beta=beta,
                lambda_p=lambda_p,
                valid=strtobool(args.valid),
                valid_data=valid_data,
                valid_loader=valid_loader,
                epochs=epochs,
                device=device,
                pad_id=pad_id,
                sep_id=sep_id,
                writer=writer,
                modelsave_pass= save_pass + args_str
            )
        else:
            print_sample_num = args.print_sample_num
            ponder_print_sample(
                x=train_data.data_x[:print_sample_num],
                true_y=train_data.data_y[:print_sample_num],
                id2word_dic=train_data.id_dic,
                model=model,
                device=device,
                pad_id=pad_id,
                sep_id=sep_id,
                load_pass=load_pass,
                
            )      

    else:
        if strtobool(args.train):
            vanilla_train(
            model=model,
            optimizer=optimizer,
            train_data=train_data,
            train_loader=train_loader,
            valid=strtobool(args.valid),
            valid_data=valid_data,
            valid_loader=valid_loader,
            epochs=epochs,
            device=device,
            pad_id=pad_id,
            sep_id=sep_id,
            writer=writer,
            modelsave_pass= save_pass + args_str
        )
        else:
            print_sample_num = args.print_sample_num
            vanilla_print_sample(
                x=train_data.data_x[:print_sample_num],
                true_y=train_data.data_y[:print_sample_num],
                id2word_dic=train_data.id_dic,
                model=model,
                device=device,
                pad_id=pad_id,
                sep_id=sep_id,
                load_pass=load_pass,
                
            )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--max_step', default=3, type=int)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--liner_dim', default=128, type=int)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--lambda_p', default=20, type=int)
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--print_sample_num', default=0, type=int)
    parser.add_argument('--valid', default='false')
    parser.add_argument('--train', default='true')
    parser.add_argument('--ponder_model', default='true')
    parser.add_argument('--lr',default=0.00003,type=float)
    parser.add_argument(
        '--load_pass', default=None)
    args = parser.parse_args()
    main(args)
