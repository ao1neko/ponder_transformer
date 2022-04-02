import argparse

import torch
import torch.nn as nn
from util import PositionalEncoder,RandomPositionalEncoder
from einops import rearrange, repeat
from loss import ReconstructionLoss, RegularizationLoss,SequentialLoss
import os

class BaseModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=512,
        num_layers=1,
        nhead=8,
        num_token=100,
        rand_pos_encoder_type = True,
        device = torch.device("cpu")
    ):
        super().__init__()
        self.num_token = num_token
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.device = device
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.rand_pos_encoder = RandomPositionalEncoder(d_model=emb_dim) if rand_pos_encoder_type == True else PositionalEncoder(d_model=emb_dim)
        self.pos_encoder = PositionalEncoder(d_model=emb_dim)
        self.output_layer = nn.Linear(emb_dim, self.num_token)
        

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        
    def encode(self, x:torch.Tensor,encoder):
        x = self.embedding(x)
        x = encoder(x)
        return x
        
    def calculate_loss(self,pred_y,true_y,pad_id=0):
        return self.loss_func(pred_y,true_y,pad_id=pad_id)
    
    def calculate_acc(self,pred_y:torch.Tensor,true_y:torch.Tensor,pad_id=0)-> int:
        count = 0
        true_y= true_y[:,1:]
        pred_y = torch.argmax(pred_y[:,:-1], dim=-1)
            
        for pred_y_sequence,true_y_sequence in zip(pred_y,true_y):
            flag = False
            for pred_y_token,true_y_token in zip(pred_y_sequence,true_y_sequence):
                if true_y_token.item() != pad_id and pred_y_token.item() != true_y_token.item(): flag = True
            if flag == False : count += 1
        return count
    
    def analyze(self,x,true_y,outputs,id_dic):
        pred_y = outputs
        with open('model_analyze/analyze.txt', 'a') as f, open('model_analyze/analyze_err.txt', 'a') as f_err:
            str_x=[id_dic[id] for id in x[0].tolist()]
            true_y_str = [id_dic[id] for id in true_y[0].tolist()]
            pred_y_str = ['<CLS>']+[id_dic[id] for id in torch.argmax(pred_y[0],dim=-1)[:-1].tolist()]
    
            compare_flag=True
            for true_y_str_w, pred_y_str_w in zip(true_y_str, pred_y_str):
                if true_y_str_w != pred_y_str_w and true_y_str_w!='<PAD>': 
                    compare_flag = False
                    break
                    
            if compare_flag == False:
                f_err.write(f'x:{str_x}\n')
                f_err.write(f'true_y:{true_y_str}\n')
                f_err.write(f'pred_y:{pred_y_str}\n') 
            else:
                f.write(f'x:{str_x}\n')
                f.write(f'true_y:{true_y_str}\n')
                f.write(f'pred_y:{pred_y_str}\n')  

    def init_analyze(self):
        file_path_list = ['model_analyze/analyze.txt','model_analyze/analyze_err.txt']
        for file_path in file_path_list:
            if os.path.exists(file_path): 
                os.remove(file_path)

    def finish_analyze(self):
        pass
    
    def output_to_input(self,pred_y:torch.Tensor)-> int:
        return torch.argmax(pred_y[:,:-1], dim=-1)
    
    
    
    
    
class Transformer(BaseModel):
    def __init__(
        self,
        vocab_size,
        emb_dim=512,
        num_layers=1,
        nhead=8,
        num_token=100,
        rand_pos_encoder_type = True,      
        device = torch.device("cpu")  
        ):
        super().__init__(
            vocab_size,
            emb_dim=emb_dim,
            num_layers=num_layers,
            nhead=nhead,
            num_token=num_token,
            rand_pos_encoder_type = rand_pos_encoder_type,  
            device = device,  
        )
        self.transformer = nn.Transformer(d_model=self.emb_dim,nhead=self.nhead,num_encoder_layers=self.num_layers,num_decoder_layers=self.num_layers,batch_first=True)
        self.init_weights()
        self.loss_func = SequentialLoss()
        
    def forward(self, x:torch.Tensor,true_y:torch.Tensor,tgt_mask:torch.Tensor,src_key_padding_mask = None,tgt_key_padding_mask=None):
        x = self.encode(x,self.rand_pos_encoder)
        true_y = self.encode(true_y,self.pos_encoder)

        y = self.transformer(x,true_y,tgt_mask=tgt_mask,src_key_padding_mask=src_key_padding_mask,tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=src_key_padding_mask)
        y = self.output_layer(y)
        return y




class DecoderOnlyTransformer(BaseModel):
    def __init__(
        self,
        vocab_size,
        emb_dim=512,
        num_layers=1,
        nhead=8,
        num_token=100,
        rand_pos_encoder_type = True,      
        device = torch.device("cpu")  
        ):
        super().__init__(
            vocab_size,
            emb_dim=emb_dim,
            num_layers=num_layers,
            nhead=nhead,
            num_token=num_token,
            rand_pos_encoder_type = rand_pos_encoder_type,  
            device = device,  
        )
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.emb_dim,nhead=self.nhead,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.num_layers)
        self.init_weights()
        self.loss_func = SequentialLoss()
        
    def forward(self, x:torch.Tensor,true_y:torch.Tensor,tgt_mask:torch.Tensor,src_key_padding_mask = None,tgt_key_padding_mask=None):
        x = self.encode(x,self.rand_pos_encoder)
        true_y = self.encode(true_y,self.pos_encoder)

        y = self.transformer_decoder(tgt=true_y,memory=x,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=src_key_padding_mask)
        y = self.output_layer(y)
        return y       

    

