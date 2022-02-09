import argparse
import numpy as np
import torch
import torch.optim as optim
from distutils.util import strtobool
import random
from torch.utils.data.dataset import Subset
from pathlib import Path

from model import PonderTransformer,Transformer,LoopTransformer
from run_model import RunModel
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from preprocess_data import  JsonDataset, make_word_dic,make_id_dic

def main(args):
    #ランダムシード設定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    
    #ハイパーパラメータ
    args_str = Path(args.comment + ':'+','.join(
            [arg+"="+str(getattr(args, arg)) for arg in vars(args) if arg not in args.ignore_comment_args]
        ))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    batch_size = args.batch_size
    beta = args.beta
    num_layers = args.num_layers
    lambda_p=args.lambda_p
    emb_dim = args.emb_dim
    model_load_path = Path(args.model_load_path)
    model_save_path = Path(args.model_save_path) / args_str
    rand_pos_encoder_type = strtobool(args.rand_pos_encoder_type)
    
    ponder_model = strtobool(args.ponder_model)
    loop_model = strtobool(args.loop_model)
    vanilla_model = strtobool(args.vanilla_model)
    assert (ponder_model + loop_model + vanilla_model) == 1, '正しいMODELを選択してください'
    
    train = strtobool(args.train)
    pretrain = strtobool(args.pretrain)
    test = strtobool(args.test)
    analyze = strtobool(args.analyze)
    
    #tensorboard
    log_dir = Path(args.tensorboard_log_dir) / args_str
    writer = SummaryWriter(log_dir=log_dir)
    #データセット
    
    
    word_dic = make_word_dic()
    id_dic = make_id_dic(word_dic)
    train_data  =  JsonDataset(args.json_base_dir,args.train_json_names,word_dic=word_dic,id_dic=id_dic)
    valid_data  =  JsonDataset(args.json_base_dir,args.valid_json_names,word_dic=word_dic,id_dic=id_dic)
    test_data  =  JsonDataset(args.json_base_dir,args.test_json_names,word_dic=word_dic,id_dic=id_dic)
    vocab_size = train_data.vocab_size
    sep_id = word_dic["<SEP>"]
    pad_id = word_dic["<PAD>"]
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    analyze_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=1,
                                               shuffle=True)
    #モデルの定義
    if ponder_model:
        model = PonderTransformer(
            vocab_size=vocab_size, 
            allow_halting=False,
            absolute_halting=strtobool(args.absolute_halting),
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_token=vocab_size,
            lambda_p = lambda_p,
            beta = beta,
            device = device,
            rand_pos_encoder_type = rand_pos_encoder_type,
            ).to(device)
    elif loop_model:
        model = LoopTransformer(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            num_token=vocab_size,
            num_layers=num_layers,
            rand_pos_encoder_type = rand_pos_encoder_type,
            device = device
            ).to(device)
    elif vanilla_model:
        model = Transformer(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            num_token=vocab_size,
            num_layers=num_layers,
            rand_pos_encoder_type = rand_pos_encoder_type,
            device = device,
            ).to(device)
    
    if strtobool(args.load_model):
        model.load_state_dict(torch.load(model_load_path)) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr,)


    #学習・評価 
    run = RunModel(
        model,
        device,
        writer=writer, 
    )
    if train:
        for epoch in tqdm(range(epochs)):
            run.train(
                optimizer=optimizer,
                train_loader=train_loader,
                epoch=epoch,
                pad_id=pad_id,
            )
            run.valid(
                valid_loader=valid_loader,
                epoch=epoch,
                pad_id=pad_id,      
                model_save_path=model_save_path,
            )
    elif pretrain:
        for epoch in tqdm(range(epochs)):
            run.train(
                optimizer=optimizer,
                train_loader=train_loader,
                epoch=epoch,
                pad_id=pad_id,
            )
            run.valid(
                valid_loader=valid_loader,
                epoch=epoch,
                pad_id=pad_id,      
                model_save_path=model_save_path,
            )
    elif test:
        run.test(
            test_loader=test_loader,
            pad_id=pad_id,
        )
    elif analyze:
         run.analyze(
            analyze_loader=analyze_loader,
            pad_id=pad_id,
        )       
        
        




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
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
    parser.add_argument('--absolute_halting', default='true')
    parser.add_argument('--rand_pos_encoder_type', default='true')
    parser.add_argument(
        '--model_load_path', default="best_models")
    parser.add_argument('--load_model', default='false') 
    parser.add_argument(
        '--model_save_path', default="best_models")
    parser.add_argument(
        '--tensorboard_log_dir', default="tensorboard_log")
    parser.add_argument(
        '--json_base_dir', default="datas")
    parser.add_argument(
        '--train_json_names',
        nargs='+',
        default=[]
        )
    parser.add_argument(
        '--valid_json_names',
        nargs='+',
        default=[]
        )
    parser.add_argument(
        '--test_json_names',
        nargs='+',
        default=[]
        )    
    parser.add_argument(
        '--comment', default="no_comment")
    parser.add_argument(
        '--ignore_comment_args', 
        nargs='+', 
        help='list of ignore args', 
        default=['model_load_path','model_save_path', 'tensorboard_log_dir','device']
        )
    args = parser.parse_args()
    main(args)
