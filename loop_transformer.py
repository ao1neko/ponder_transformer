import torch
import torch.nn as nn
from util import make_tgt_mask, PositionalEncoder



class LoopTransformerGenerater(nn.Module):
    def __init__(
        self,vocab_size ,emb_dim=128,num_layers=1,nhead=8,num_token=100
    ):
        """Transformer

        Args:
            vocab_size (int):
            emb_dim (int, optional): Defaults to 300.
            num_layers (int, optional): Defaults to 1.
            nhead (int, optional): Defaults to 10.
            num_token: kind of output tokens.
        """
        super().__init__()
        self.num_token = num_token
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.pos_encoder = PositionalEncoder(d_model=emb_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        self.output_layer = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_token)
        )

        
    def forward(self, x:torch.Tensor,src_key_padding_mask = None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for _ in range(self.num_layers):
            x = self.transformer_encoder(x,src_key_padding_mask=src_key_padding_mask)
        y = self.output_layer(x[:,0])
        return y
    
    
    

class LoopTransformer(nn.Module):
    def __init__(
        self,vocab_size ,emb_dim=512, max_steps=6, allow_halting=False,nhead=8,num_token=100,liner_dim=128
    ):
        """ PonderNet + Transformer

        Args:
            vocab_size (int): 
            emb_dim (int, optional): Defaults to 300.
            max_steps (int, optional): Defaults to 20.
            allow_halting (bool, optional): If True, then the forward pass is allowed to halt before
                                            reaching the maximum steps. Defaults to False.
            nhead (int, optional): Defaults to 10.
            num_token: kind of output tokens
        """
        super().__init__()
        self.max_steps = max_steps
        self.allow_halting = allow_halting
        self.num_token = num_token
        self.liner_dim = liner_dim 
        self.emb_dim = emb_dim
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.pos_encoder = PositionalEncoder(d_model=emb_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=1)
        self.output_layer = nn.Linear(emb_dim, self.num_token)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x:torch.Tensor,true_y:torch.Tensor,tgt_mask:torch.Tensor=None,src_key_padding_mask = None,tgt_key_padding_mask=None):
        """
        Args:
            x (torch.Tensor): input. shape of x is (batch_size, seq_len)
            true_y (torch.Tensor): label
            tgt_mask (torch.Tensor, optional): target_mask. Defaults to None.

        Returns:
            y (torch.Tensor): output. shape of y is (max,step, batch_size, seq_len, dim)
            p (torch.Tensor): halt probability. shape of p is (max_step, batch_size)
            halting_step (torch.Tensor): num of loop. shape of halting_step is (batch_size)
        """
        
        x = self.embedding(x)
        x = self.pos_encoder(x)
        true_y = self.embedding(true_y)
        true_y = self.pos_encoder(true_y)

        for _ in range(self.max_steps):
            x = self.transformer_encoder(x,src_key_padding_mask=src_key_padding_mask)    
        h = true_y.clone()
        
        for _ in range(self.max_steps):
            h = self.transformer_decoder(tgt=h,memory=x,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=src_key_padding_mask)
        h = self.output_layer(h)
        return h