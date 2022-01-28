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
from loop_transformer import LoopTransformer
from datasets import ConcatedMultiReasoningBERTData


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
                            for arg in vars(args) if arg not in ['json_pass', 'load_pass','save_pass', 'log_dir']])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    batch_size = args.batch_size
    beta = args.beta
    ponder_model = strtobool(args.ponder_model)
    loop_model = strtobool(args.loop_model)
    
    log_dir = args.log_dir + args_str
        
    writer = SummaryWriter(log_dir=log_dir, comment=args_str)
    
    pass_list = ["depth1.json"]
    all_data = ConcatedMultiReasoningBERTData([args.json_pass + x for x in pass_list])
    
    all_data_len = len(all_data)  # n_samples is 60000
    train_len = int(all_data_len * 0.8)
    train_indices = list(range(0, train_len))  # [0,1,.....47999]
    valid_indices = list(range(train_len, all_data_len))  # [48000,48001,.....59999]
    train_data = Subset(all_data, train_indices)
    valid_data = Subset(all_data, valid_indices)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    vocab_size = all_data.vocab_size
    max_step = args.max_step
    lambda_p=args.lambda_p
    emb_dim = args.emb_dim
    liner_dim = args.liner_dim
    load_pass = args.load_pass
    save_pass = args.save_pass

    if ponder_model:
        model = PonderTransformer(vocab_size=vocab_size, allow_halting=False,emb_dim=emb_dim,
                                  max_steps=max_step, num_token=vocab_size,liner_dim=liner_dim).to(device)
    elif loop_model:
        model = LoopTransformer(vocab_size=vocab_size,emb_dim=emb_dim,
                            num_token=vocab_size, max_steps=max_step).to(device)
    else:
        model = Transformer(vocab_size=vocab_size,emb_dim=emb_dim,
                            num_token=vocab_size, num_layers=max_step,liner_dim=liner_dim).to(device)
    
    model.load_state_dict(torch.load(load_pass)) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr,)

    sep_id = all_data.word_dic["<SEP>"]
    pad_id = all_data.word_dic["<PAD>"]


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
        if strtobool(args.test):
            ponder_test(
                model=model,
                test_data=valid_data,
                test_loader=valid_loader,
                max_step=max_step,
                device=device,
                pad_id=pad_id,
                sep_id=sep_id,
                id_dic=all_data.id_dic
            )
            print_sample_num = args.print_sample_num
            ponder_print_sample(
                x=all_data.data_x[-print_sample_num:],
                true_y=all_data.data_y[-print_sample_num:],
                id2word_dic=all_data.id_dic,
                model=model,
                device=device,
                pad_id=pad_id,
                sep_id=sep_id,
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
        if strtobool(args.test):
            vanilla_test(
                model=model,
                test_data=valid_data,
                test_loader=valid_loader,
                device=device,
                pad_id=pad_id,
                sep_id=sep_id
            )
            print_sample_num = args.print_sample_num
            vanilla_print_sample(
                x=all_data.data_x[:print_sample_num],
                true_y=all_data.data_y[:print_sample_num],
                id2word_dic=all_data.id_dic,
                model=model,
                device=device,
                pad_id=pad_id,
                sep_id=sep_id,
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
    parser.add_argument('--train', default='true')
    parser.add_argument('--test', default='false')
    parser.add_argument('--valid', default='true')
    parser.add_argument('--ponder_model', default='true')
    parser.add_argument('--loop_model', default='false')
    parser.add_argument('--lr',default=0.00003,type=float)
    parser.add_argument(
        '--load_pass', default=None)
    parser.add_argument(
        '--json_pass', default=None)
    parser.add_argument(
        '--save_pass', default=None)
    parser.add_argument(
        '--log_dir', default=None)
    args = parser.parse_args()
    main(args)
