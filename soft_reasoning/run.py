import argparse
import numpy as np
import torch
import torch.optim as optim
from distutils.util import strtobool
import random
import numpy as np
import datetime
import sys
import os
sys.path.append(os.pardir)

from ponder_transformer import PonderTransformerClassifier
from vanilla_transformer import TransformerrClassifier
from util import make_word_dic, make_id_dic
from datasets import SoftReasonersData
import torch.nn.functional as F
import json

from torch.utils.tensorboard import SummaryWriter
from run_ponder_transformer import ponder_train, ponder_test, ponder_print_sample
from run_vanilla_transformer import vanilla_train, vanilla_test, vanilla_print_sample


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    daytime = str(datetime.datetime.fromtimestamp(0))
    args_str = '_'+','.join([arg+"="+str(getattr(args, arg))
                            for arg in vars(args) if arg not in ['data_pass', 'load_pass', 'log_dir']])
    log_dir = args.log_dir
    if log_dir is not None:
        log_dir = log_dir + '/args' + args_str
    writer = SummaryWriter(log_dir=log_dir, comment=args_str)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(device)
    epochs = args.epochs
    batch_size = args.batch_size
    beta = args.beta
    ponder_model = strtobool(args.ponder_model)

    train_data = SoftReasonersData(args.data_pass + "/train.jsonl")
    valid_data = SoftReasonersData(args.data_pass+"/dev.jsonl")
    save_pass = "/home/aoki0903/sftp_sync/MyPonderNet/nlp_datasets/rule-reasoning-dataset-V2020.2.5.0/"
    make_word_dic([train_data.data_x, valid_data.data_x], save_pass,
                  init_word_dic={"<PAD>": 0, "true": 1, "false": 2})
    with open(save_pass + "word_dic.json", "r") as tf:
        word_dic = json.load(tf)
        # print(word_dic.items())

    make_id_dic(word_dic, save_pass)
    with open(save_pass + "id_dic.json", "r") as tf:
        id_dic = json.load(tf)
        id_dic = {int(k): v for k, v in id_dic.items()}
        # print(id_dic.items())
    
    train_data.def_dic(word_dic,id_dic)
    valid_data.def_dic(word_dic,id_dic)
    train_data.text2id(pad_id=0)
    valid_data.text2id(pad_id=0)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    vocab_size = len(train_data.word_dic)
    max_step = args.max_step

    if ponder_model:
        model = PonderTransformerClassifier(vocab_size=vocab_size, allow_halting=False,
                                            max_steps=max_step, num_token=2).to(device)
    else:
        model = TransformerrClassifier(
            vocab_size=vocab_size, num_token=2, num_layers=max_step).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0003,)

    sep_id = train_data.word_dic["<SEP>"]
    pad_id = train_data.word_dic["<PAD>"]

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
                valid=strtobool(args.valid),
                valid_data=valid_data,
                valid_loader=valid_loader,
                epochs=epochs,
                device=device,
                pad_id=pad_id,
                sep_id=sep_id,
                writer=writer,
                modelsave_pass='best_ponder_models/' + daytime + args_str
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
                sep_id=sep_id
            )

        if args.print_sample_num > 0:
            print_sample_num = args.print_sample_num
            ponder_print_sample(
                x=train_data.id_data_x[:print_sample_num],
                true_y=train_data.id_data_y[:print_sample_num],
                id2word_dic=train_data.id_dic,
                model=model,
                device=device,
                pad_id=pad_id,
                sep_id=sep_id
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
                modelsave_pass='best_vanilla_models/' + daytime + args_str
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
                x=train_data.id_data_x[:print_sample_num],
                true_y=train_data.id_data_y[:print_sample_num],
                id2word_dic=train_data.id_dic,
                model=model,
                device=device,
                pad_id=pad_id,
                sep_id=sep_id
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--max_step', default=3, type=int)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument(
        '--data_pass', default="/home/aoki0903/sftp_sync/MyPonderNet/nlp_datasets/rule-reasoning-dataset-V2020.2.5.0/original/depth-3ext")
    parser.add_argument(
        '--load_pass', default=None)
    parser.add_argument(
        '--log_dir', default=None)
    parser.add_argument('--train', default='true')
    parser.add_argument('--test', default='false')
    parser.add_argument('--valid', default='false')
    parser.add_argument('--print_sample_num', default=0, type=int)
    parser.add_argument('--ponder_model', default='true')
    args = parser.parse_args()
    main(args)
