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

from model import PonderTransformer,Transformer,LoopTransformer
from datasets import ConcatedMultiReasoningBERTData
from preprocess_data import  JsonDataset, SimpleDataset
from run_model import RunModel
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter


def main(args):
    #ランダムシード設定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    #ハイパーパラメータ
    args_str = ':'+','.join(
            [arg+"="+str(getattr(args, arg)) for arg in vars(args) if arg not in args.ignore_comment_args]
        )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    batch_size = args.batch_size
    beta = args.beta
    max_step = args.max_step
    lambda_p=args.lambda_p
    emb_dim = args.emb_dim
    model_load_path = args.model_load_path
    model_save_path = args.model_save_path
    ponder_model = strtobool(args.ponder_model)
    loop_model = strtobool(args.loop_model)
    vanilla_model = strtobool(args.vanilla_model)
    assert (ponder_model + loop_model + vanilla_model) == 1, '正しいMODELを選択してください'
    
    train = strtobool(args.train)
    pretrain = strtobool(args.pretrain)
    test = strtobool(args.test)
    analyze = strtobool(args.analyze)
    
    #tensorboard
    log_dir = args.log_dir + args_str
    writer = SummaryWriter(log_dir=log_dir, comment=args_str)
    
    #データセット
    all_data  =  JsonDataset(args.json_base_dir,args.json_names)
    vocab_size = all_data.vocab_size
    sep_id = all_data.word_dic["<SEP>"]
    pad_id = all_data.word_dic["<PAD>"]
    
    train_data = SimpleDataset(all_data.train_data_x,all_data.train_data_y)
    valid_data = SimpleDataset(all_data.valid_data_x,all_data.valid_data_y)
    test_data = SimpleDataset(all_data.test_data_x,all_data.test_data_y)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    analyze_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                               batch_size=1,
                                               shuffle=True)
    #モデルの定義
    if ponder_model:
        model = PonderTransformer(
            vocab_size=vocab_size, 
            allow_halting=False,
            emb_dim=emb_dim,
            max_steps=max_step,
            num_token=vocab_size
            ).to(device)
    elif loop_model:
        model = LoopTransformer(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            num_token=vocab_size,
            max_steps=max_step
            ).to(device)
    elif vanilla_model:
        model = Transformer(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            num_token=vocab_size,
            num_layers=max_step
            ).to(device)
    
    if train or test:
        model.load_state_dict(torch.load(model_load_path)) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr,)


    #学習・評価 
    run = RunModel(
        model,
        device,
        pad_id=pad_id,
        writer=writer, 
    )
    if train:
        for epoch in tqdm(range(epochs)):
            run.train(
                optimizer,
                train_loader,
                epoch,
                pad_id,
            )
            run.valid(
                valid_loader,
                epoch,
                pad_id,      
                model_save_path,
            )
    elif pretrain:
        for epoch in tqdm(range(epochs)):
            run.train(
                optimizer,
                train_loader,
                epoch,
                pad_id,
            )
    elif test:
        run.test(
            test_loader,
            pad_id,
        )
    elif analyze:
         run.analyze(
            analyze_loader,
            pad_id,
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
    parser.add_argument('--train', default='true')
    parser.add_argument('--pretrain', default='false')
    parser.add_argument('--test', default='false')
    parser.add_argument('--analyze', default='false')
    parser.add_argument('--ponder_model', default='true')
    parser.add_argument('--loop_model', default='false')
    parser.add_argument('--vanilla_model', default='false')
    parser.add_argument('--lr',default=0.0003,type=float)
    parser.add_argument(
        '--model_load_path', default=None)
    parser.add_argument(
        '--model_save_path', default=None)
    parser.add_argument(
        '--tensorboard_log_dir', default=None)
    parser.add_argument(
        '--json_base_dir', default=None)
    parser.add_argument(
        '--json_names',
        nargs='+',
        default=['model_load_pass','model_save_pass', 'tensorboard_log_dir','device']
        )
    parser.add_argument(
        '--comment', default=None)
    parser.add_argument(
        '--ignore_comment_args', 
        nargs='+', 
        help='list of ignore args', 
        default=['model_load_path','model_save_path', 'tensorboard_log_dir','device']
        )
    args = parser.parse_args()
    main(args)
