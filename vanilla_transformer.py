import argparse

import torch
import torch.nn as nn
from util import make_tgt_mask

class Transformer(nn.Module):
    def __init__(
        self,vocab_size ,emb_dim=300,num_layers=1,nhead=10
    ):
        """Transformer

        Args:
            vocab_size (int):
            emb_dim (int, optional): Defaults to 300.
            num_layers (int, optional): Defaults to 1.
            nhead (int, optional): Defaults to 10.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.transformer = nn.Transformer(d_model=emb_dim,nhead=nhead,num_encoder_layers=num_layers,num_decoder_layers=num_layers,batch_first=True)

        
    def forward(self, x,true_y,tgt_mask):
        src_key_padding_mask = (x == 0)
        tgt_key_padding_mask = (true_y == 0)
        
        x = self.embedding(x)
        true_y = self.embedding(true_y)   
        y = self.transformer(x,true_y,tgt_mask=tgt_mask,src_key_padding_mask=src_key_padding_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        return y


def main():
    vocab_size = 500
    model = Transformer(vocab_size=vocab_size,num_layers=1)
    x = torch.randint(low=0,high=vocab_size-1,size=(2,50))
    true_y = torch.randint(low=0,high=vocab_size-1,size=(2,5))
    tgt_mask = make_tgt_mask(5)
    pred_y = model(x,true_y,tgt_mask=tgt_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明') 
    parser.add_argument('--arg1', help='この引数の説明') # 必須の引数を追加
    args = parser.parse_args()
    main()
