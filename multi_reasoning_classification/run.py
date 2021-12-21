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

from ponder_transformer import PonderTransformerGenerater
from vanilla_transformer import TransformerGenerater
from loop_transformer import LoopTransformerGenerater
from datasets import MultiReasoningData,ConcatedMultiReasoningData


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
    #torch.set_printoptions(edgeitems=10)
    args_str = '_'+','.join([arg+"="+str(getattr(args, arg))
                            for arg in vars(args) if arg not in ['json_pass', 'load_pass', 'log_dir']])
    log_dir = args.log_dir
    if log_dir is not None:
        log_dir = log_dir + '/args' + args_str
    writer = SummaryWriter(log_dir=log_dir, comment=args_str)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    epochs = args.epochs
    batch_size = args.batch_size
    beta = args.beta
    ponder_model = strtobool(args.ponder_model)
    loop_model = strtobool(args.loop_model)
    concated = strtobool(args.concated)
    if concated:
        pass_list = ["mod_depth2.json","mod_depth3.json","mod_depth4.json","mod_depth5.json"]
        all_data = ConcatedMultiReasoningData([args.json_pass + x for x in pass_list])
    else:
        all_data = MultiReasoningData(args.json_pass)  # if文で切り替える?,testはこれを分割して使用
    
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

    if ponder_model:
        model = PonderTransformerGenerater(vocab_size=vocab_size, allow_halting=False,emb_dim=emb_dim,
                                  max_steps=max_step, num_token=vocab_size).to(device)
    elif loop_model:
        model = LoopTransformerGenerater(vocab_size=vocab_size,emb_dim=emb_dim,
                            num_token=vocab_size, num_layers=max_step).to(device)
    else:
        model = TransformerGenerater(vocab_size=vocab_size,emb_dim=emb_dim,
                            num_token=vocab_size, num_layers=max_step).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,)

    sep_id = all_data.word_dic["<SEP>"]
    pad_id = all_data.word_dic["<PAD>"]

    load_pass = args.load_pass

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
                modelsave_pass= load_pass + args_str
            )

        if strtobool(args.test):
            ponder_test(
                model=model,
                test_data=valid_data,
                test_loader=valid_loader,
                max_step=max_step,
                device=device,
                load_pass=load_pass,
                pad_id=pad_id,
                sep_id=sep_id,
                concated = concated
            )
        if args.print_sample_num > 0:
            print_sample_num = args.print_sample_num
            ponder_print_sample(
                x=all_data.data_x[:print_sample_num],
                true_y=all_data.data_y[:print_sample_num],
                id2word_dic=all_data.id_dic,
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
                modelsave_pass=load_pass+ args_str
            )

        if strtobool(args.test):
            vanilla_test(
                model=model,
                test_data=valid_data,
                test_loader=valid_loader,
                device=device,
                load_pass=load_pass,
                pad_id=pad_id,
                sep_id=sep_id
            )

        if args.print_sample_num > 0:
            print_sample_num = args.print_sample_num
            vanilla_print_sample(
                x=all_data.data_x[:print_sample_num],
                true_y=all_data.data_y[:print_sample_num],
                id2word_dic=all_data.id_dic,
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
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--lambda_p', default=20, type=int)
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument(
        '--json_pass', default="/work01/aoki0903/PonderNet/multihop_experiment/datas/ponder_base.json")
    parser.add_argument(
        '--load_pass', default=None)
    parser.add_argument(
        '--log_dir', default=None)
    parser.add_argument('--train', default='true')
    parser.add_argument('--test', default='false')
    parser.add_argument('--valid', default='false')
    parser.add_argument('--print_sample_num', default=0, type=int)
    parser.add_argument('--ponder_model', default='true')
    parser.add_argument('--concated', default='false')
    parser.add_argument('--loop_model', default='false')
    parser.add_argument('--lr',default=0.00003,type=float)
    args = parser.parse_args()
    main(args)
