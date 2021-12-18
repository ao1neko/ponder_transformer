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