import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import sys
import os
from einops import rearrange, repeat

sys.path.append(os.pardir)
from vanilla_transformer import ReconstructionLoss

from util import make_tgt_mask
from torch.utils.tensorboard import SummaryWriter

from datasets import SingleReasoningBERTData
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout,batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model,nhead,d_hid, dropout,batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src,tgt, src_mask,tgt_mask,tgt_key_padding_mask) :
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = rearrange(src, 'b s-> s b')
        tgt = rearrange(tgt, 'b s-> s b')
        
        src = self.encoder(src) * math.sqrt(self.d_model)
        tgt = self.encoder(tgt) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        src = rearrange(src, 's b d-> b s d')
        tgt = rearrange(tgt, 's b d-> b s d')

        memory = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = self.transformer_decoder(tgt,memory, tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        
        output = self.decoder(output)
        return output


def calculate_acc(pred_y:torch.Tensor,true_y:torch.Tensor,pad_id=0)-> int:
    """
    Args:
        pred_y (torch.Tensor):output. shape of y is (batch_size, seq_len, dim)
        true_y (torch.Tensor):
        h (torch.Tensor) :
    Returns:
        int: num of accuracy
    """
  
    count = 0
    true_y= true_y[:,1:]
    pred_y = torch.argmax(pred_y[:,:-1], dim=-1)
        
    for pred_y_sequence,true_y_sequence in zip(pred_y,true_y):
        flag = False
        for pred_y_token,true_y_token in zip(pred_y_sequence,true_y_sequence):
            if true_y_token.item() != pad_id and pred_y_token.item() != true_y_token.item(): flag = True
        if flag == False : count += 1

    return count

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x) :
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
    
def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    train_data = SingleReasoningBERTData() 
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=128,
                                               shuffle=True)
    vocab_size = train_data.vocab_size
    
    ntokens = vocab_size  # size of vocabulary
    emsize = 128  # embedding dimension
    d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 6  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 8  # number of heads in nn.MultiheadAttention
    dropout = 0.2  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    

    loss_rec_inst = ReconstructionLoss()
    for epoch in tqdm(range(150)):
        total_loss = 0.0
        total_acc = 0.0

        model.train()
        for x, true_y in train_loader:
            x = x.to(device)
            true_y = true_y.to(device)
            tgt_mask = make_tgt_mask(true_y.shape[1]).to(device)
            src_key_padding_mask = (x == 0)
            tgt_key_padding_mask = (true_y == 0)
            
            
            optimizer.zero_grad()
            pred_y = model(x, true_y, tgt_mask=tgt_mask,
                           src_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            loss = loss_rec_inst(pred_y,true_y,pad_id=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            total_acc += calculate_acc(pred_y, true_y,pad_id=0)
        print(f"train_loss:{total_loss/len(train_loader)}")
        print(f"train_acc:{total_acc/len(train_data)}")



if __name__ == '__main__':
    main()