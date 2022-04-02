import argparse
import numpy as np
import torch
import torch.optim as optim
from distutils.util import strtobool
import random
from pathlib import Path

from model import Transformer,DecoderOnlyTransformer
from run_model import RunVanillaModel,RunDecoderModel,RunDecoderMultiStepModel

from torch.utils.tensorboard import SummaryWriter
from preprocess_data import  OnestepDataset, make_word_dic,make_id_dic

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
    num_layers = args.num_layers
    emb_dim = args.emb_dim
    model_load_path = Path(args.model_load_path)
    model_save_path = Path(args.model_save_path) / args_str
    rand_pos_encoder_type = strtobool(args.rand_pos_encoder_type)
    vanilla_model = strtobool(args.vanilla_model)
    decoder_model = strtobool(args.decoder_model)
    decoder_multi_model = strtobool(args.decoder_multi_model)
    assert (decoder_model + vanilla_model + decoder_multi_model) == 1, '正しいMODELを選択してください'
    
    train = strtobool(args.train)
    pretrain = strtobool(args.pretrain)
    test = strtobool(args.test)
    analyze = strtobool(args.analyze)
    
    #tensorboard
    log_dir = Path(args.tensorboard_log_dir) / args_str
    writer = SummaryWriter(log_dir=log_dir) if train or pretrain else None
    
    #データセット
    train_data  =  OnestepDataset(args.json_base_dir,args.train_json_names)
    valid_data  =  OnestepDataset(args.json_base_dir,args.valid_json_names)
    test_data  =  OnestepDataset(args.json_base_dir,args.test_json_names)
    vocab_size = train_data.vocab_size
    pad_id = train_data.word_dic["<PAD>"]

    if vanilla_model:
        model = Transformer(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            num_token=vocab_size,
            num_layers=num_layers,
            rand_pos_encoder_type = rand_pos_encoder_type,
            device = device,
            ).to(device)
        #学習・評価 
        run = RunVanillaModel(
            model,
            device,
            writer=writer, 
            word_dic=train_data.word_dic,
            id_dic=train_data.id_dic,
            batch_size=batch_size,
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
        )
    elif decoder_model:
        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            num_token=vocab_size,
            num_layers=num_layers,
            rand_pos_encoder_type = rand_pos_encoder_type,
            device = device,
            ).to(device)
        #学習・評価 
        run = RunDecoderModel(
            model,
            device,
            writer=writer, 
            word_dic=train_data.word_dic,
            id_dic=train_data.id_dic,
            batch_size=batch_size,
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
        )
    elif decoder_multi_model:
        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            num_token=vocab_size,
            num_layers=num_layers,
            rand_pos_encoder_type = rand_pos_encoder_type,
            device = device,
            ).to(device)
        #学習・評価 
        run = RunDecoderMultiStepModel(
            model,
            device,
            writer=writer, 
            word_dic=train_data.word_dic,
            id_dic=train_data.id_dic,
            batch_size=batch_size,
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
        )        
                
    if strtobool(args.load_model):
        model.load_state_dict(torch.load(model_load_path)) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr,)






    if train:
        run.train(
            optimizer=optimizer,
            epochs=epochs,
            pad_id=pad_id,
            model_save_path=model_save_path,
        )
    elif pretrain:
        run.train(
            optimizer=optimizer,
            epochs=epochs,
            pad_id=pad_id,
            model_save_path=model_save_path,
        )
    elif test:
        run.test(
            pad_id=pad_id,
        )
    elif analyze:
         run.analyze(
            pad_id=pad_id,
        )       
        
        




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_layers', default=6, type=int)
    parser.add_argument('--seed', default=6, type=int)
    parser.add_argument('--emb_dim', default=512, type=int)
    parser.add_argument('--device', default="cuda:0")
    parser.add_argument('--train', default='false')
    parser.add_argument('--pretrain', default='false')
    parser.add_argument('--test', default='false')
    parser.add_argument('--analyze', default='false')
    parser.add_argument('--vanilla_model', default='false')
    parser.add_argument('--decoder_model', default='false')
    parser.add_argument('--decoder_multi_model', default='false')
    parser.add_argument('--lr',default=0.0001,type=float)
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
