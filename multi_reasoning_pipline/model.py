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
        
    def calculate_loss(self):
        pass    
    
    def calculate_acc(self):
        pass 
    
    def analyze(self):
        pass     



class PonderTransformer(BaseModel):
    def __init__(
        self,
        vocab_size,
        emb_dim=512,
        num_layers=1,
        nhead=8,
        num_token=100,
        rand_pos_encoder_type = True,
        allow_halting=False,
        absolute_halting = False,
        lambda_p = 6,
        beta = 0.01,
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
        self.allow_halting = allow_halting
        self.absolute_halting = absolute_halting
        self.lambda_p = lambda_p
        self.beta = beta
        
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim,nhead=self.nhead,batch_first=True)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.emb_dim,nhead=self.nhead,batch_first=True)
        self.norm = nn.LayerNorm(self.emb_dim,eps = 1e-5)
        
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=1)
        self.lambda_layer = nn.Linear(self.emb_dim, 1)
        self.init_weights()
        
        self.loss_rec_inst = ReconstructionLoss()
        self.loss_reg_inst = RegularizationLoss(lambda_p=1.0/lambda_p, num_layers=self.num_layers, device=self.device)

    def init_weights(self) -> None:
        super().init_weights()
        initrange = 0.1
        self.lambda_layer.bias.data.zero_()
        self.lambda_layer.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x:torch.Tensor,true_y:torch.Tensor,tgt_mask:torch.Tensor=None,src_key_padding_mask = None,tgt_key_padding_mask=None):
        x = self.encode(x,self.rand_pos_encoder)
        true_y = self.encode(true_y,self.pos_encoder)

        batch_size, _, _ = x.shape
        device = x.device
        un_halted_prob = x.new_ones(batch_size)

        y_list = []
        p_list = []

        halting_step = torch.zeros(
            batch_size,
            dtype=torch.long,
            device=device,
        )
        for n in range(1, self.num_layers + 1):
            x = self.transformer_encoder(x,src_key_padding_mask=src_key_padding_mask)
            
            if n == self.num_layers:
                lambda_n = x.new_ones(batch_size)  # (batch_size,)
            else:
                lambda_n = torch.sigmoid(self.lambda_layer(x[:,0]))[:, 0]  # (batch_size,)
            
            # Store releavant outputs
            #y_list.append(self.output_layer(x))  # (batch_size,)
            h = true_y.clone()
            for _ in range(n):
                h = self.transformer_decoder(tgt=h,memory=x,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=src_key_padding_mask)
            h = self.norm(h)
            h = self.output_layer(h)
            h = self.norm(h)
            y_list.append(h)
            
            
            
            p_list.append(un_halted_prob * lambda_n)  # (batch_size,)
            
            if not self.training and self.absolute_halting == True:
                p_tensor = torch.stack(p_list)
                halting_step = torch.maximum(n*(halting_step == 0)*(p_tensor.sum(dim=0) > 0.5).to(torch.long),halting_step,)
            else:
                halting_step = torch.maximum(n*(halting_step == 0)*torch.bernoulli(lambda_n).to(torch.long),halting_step,)
            
            # Prepare for next iteration
            un_halted_prob = un_halted_prob * (1 - lambda_n)

            # Potentially stop if all samples halted
            if self.allow_halting and (halting_step > 0).sum() == batch_size:
                break
        
        y = torch.stack(y_list)
        p = torch.stack(p_list)

        return (y, p, halting_step)
    
    def calculate_loss(self,outputs,true_y,pad_id=0):
        pred_y, p, halting_step = outputs
        loss_rec = self.loss_rec_inst(p, pred_y, true_y, pad_id=pad_id)
        loss_reg = self.loss_reg_inst(p,)
        loss_overall = loss_rec + self.beta * loss_reg
        return loss_overall
    
    def calculate_acc(self,outputs,true_y,pad_id=0)-> int:
        pred_y, p, h = outputs
        count = 0
        pred_y = rearrange(pred_y, 'n b s d-> b n s d')

        for pred_y_n,true_y_sequence,h_item in zip(pred_y,true_y,h):
            pred_y_sequence = pred_y_n[h_item.item()-1]
            
            true_y_sequence = true_y_sequence[1:]
            pred_y_sequence = torch.argmax(pred_y_sequence[:-1], dim=-1)
            
            flag = False
            for pred_y_token,true_y_token in zip(pred_y_sequence,true_y_sequence):
                if true_y_token.item() != pad_id and pred_y_token.item() != true_y_token.item(): flag = True
            if flag == False : count += 1
        return count
        
    def _flager(self,x,id_dic):
        #足し算の回数でフラッグ
        x = "".join([id_dic[x_id.item()] for x_id in x[0]])
        x = x.replace("<PAD>","")
        x = x[5:-5]
        return len(x.split("+")) - 1    
    
    def init_analyze(self):
        file_path = 'model_analyze/analyze.txt'
        if os.path.exists(file_path): 
            os.remove(file_path)
        self.matrix = torch.zeros(4,self.num_layers+1).to(self.device)
    
    def analyze(self,x,true_y,outputs,test_loader):
        id_dic = test_loader.dataset.id_dic
        pred_y, p, h = outputs
        with open('model_analyze/analyze.txt', 'a') as f:
            str_x=[id_dic[id] for id in x[0].tolist()]
            true_y_str = [id_dic[id] for id in true_y[0].tolist()]
            pred_y_str = ['<CLS>']+[id_dic[id] for id in torch.argmax(pred_y[h[0]-1][0],dim=-1)[:-1].tolist()]
            halting_step_str = h[0]
            halting_probability_str = p[0]
            f.write(f'x:{str_x}\n')
            f.write(f'true_y:{true_y_str}\n')
            f.write(f'pred_y:{pred_y_str}\n')
            f.write(f'halting_step:{halting_step_str}\n')
            f.write(f'halting_probability:{halting_probability_str}\n')
        flag = self._flager(x,id_dic)
        self.matrix[flag][h-1] += 1

    def finish_analyze(self):
        print(self.matrix)
        
    
    
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


    def calculate_loss(self,pred_y,true_y,pad_id=0):
        return self.loss_func(pred_y,true_y,pad_id=pad_id)
        

    def init_analyze(self):
        file_path_list = ['model_analyze/analyze.txt','model_analyze/analyze_err.txt']
        for file_path in file_path_list:
            if os.path.exists(file_path): 
                os.remove(file_path)
    
    def analyze(self,x,true_y,outputs,test_loader):
        id_dic = test_loader.dataset.id_dic
        pred_y = outputs
        with open('model_analyze/analyze.txt', 'a') as f, open('model_analyze/analyze_err.txt', 'a') as f_err:
            str_x=[id_dic[id] for id in x[0].tolist()]
            true_y_str = [id_dic[id] for id in true_y[0].tolist()]
            pred_y_str = ['<CLS>']+[id_dic[id] for id in torch.argmax(pred_y[0],dim=-1)[:-1].tolist()]
            f.write(f'x:{str_x}\n')
            f.write(f'true_y:{true_y_str}\n')
            f.write(f'pred_y:{pred_y_str}\n')
            
            compare_flag=True
            for true_y_str_w, pred_y_str_w in zip(true_y_str, pred_y_str):
                if true_y_str_w != pred_y_str_w and pred_y_str_w!='<SEP>': 
                    compare_flag = False
                    break
                    
            if compare_flag == False:
                f_err.write(f'x:{str_x}\n')
                f_err.write(f'true_y:{true_y_str}\n')
                f_err.write(f'pred_y:{pred_y_str}\n') 

    def finish_analyze(self):
        pass
        

class LoopTransformer(BaseModel):
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
            device = device
        )
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim,nhead=self.nhead,batch_first=True)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.emb_dim,nhead=self.nhead,batch_first=True)
        self.norm = nn.LayerNorm(self.emb_dim,eps = 1e-5)
       
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=1)
        self.init_weights()
        self.calculate_acc = Transformer().calculate_acc()
        
    def forward(self, x:torch.Tensor,true_y:torch.Tensor,tgt_mask:torch.Tensor=None,src_key_padding_mask = None,tgt_key_padding_mask=None):
        x = self.encode(x,self.rand_pos_encoder)
        true_y = self.encode(true_y,self.pos_encoder)

        for _ in range(self.num_layers):
            x = self.transformer_encoder(x,src_key_padding_mask=src_key_padding_mask)    
        h = true_y.clone()
        h = self.norm(h)
        
        for _ in range(self.num_layers):
            h = self.transformer_decoder(tgt=h,memory=x,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)
        h = self.output_layer(h)
        return h

#後で
    def calculate_acc(pred_y:torch.Tensor,true_y:torch.Tensor,pad_id=0)-> int:
        return Transformer().calculate_acc(pred_y,true_y,pad_id=0)
    
    def calculate_loss(self,pred_y,true_y,pad_id=0):
        return Transformer().calculate_loss(pred_y,true_y,pad_id=pad_id)