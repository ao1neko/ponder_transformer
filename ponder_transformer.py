import argparse

import torch
import torch.nn as nn
from util import make_tgt_mask


class PonderTransformer(nn.Module):
    def __init__(
        self,vocab_size ,emb_dim=300, max_steps=20, allow_halting=False,nhead=10
    ):
        """ PonderNet + Transformer

        Args:
            vocab_size (int): 
            emb_dim (int, optional): Defaults to 300.
            max_steps (int, optional): Defaults to 20.
            allow_halting (bool, optional): If True, then the forward pass is allowed to halt before
                                            reaching the maximum steps. Defaults to False.
            nhead (int, optional): Defaults to 10.
        """
        super().__init__()
        self.max_steps = max_steps
        self.allow_halting = allow_halting
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.lambda_layer = nn.Linear(emb_dim, 1)

        
    def forward(self, x,true_y,tgt_mask=None):
        src_key_padding_mask = (x == 0)
        tgt_key_padding_mask = (true_y == 0)
        x = self.embedding(x)
        true_y = self.embedding(true_y)
        
        
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

        for n in range(1, self.max_steps + 1):
            x = self.transformer_encoder_layer(x,src_key_padding_mask=src_key_padding_mask)
            if n == self.max_steps:
                lambda_n = x.new_ones(batch_size)  # (batch_size,)
            else:
                
                lambda_n = torch.sigmoid(self.lambda_layer(x[:,0]))[
                    :, 0
                ]  # (batch_size,)
            
            # Store releavant outputs
            #y_list.append(self.output_layer(x))  # (batch_size,)
            h = true_y.clone()
            for _ in range(n):
                print(h.shape)
                print(tgt_key_padding_mask.shape)
                h = self.transformer_decoder_layer(tgt=h,memory=x,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
            y_list.append(h)
            
            
            
            p_list.append(un_halted_prob * lambda_n)  # (batch_size,)

            halting_step = torch.maximum(
                n
                * (halting_step == 0)
                * torch.bernoulli(lambda_n).to(torch.long),
                halting_step,
            )

            # Prepare for next iteration
            un_halted_prob = un_halted_prob * (1 - lambda_n)

            # Potentially stop if all samples halted
            if self.allow_halting and (halting_step > 0).sum() == batch_size:
                break

        y = torch.stack(y_list)
        p = torch.stack(p_list)

        return y, p, halting_step


def main():
    vocab_size = 500
    model = PonderTransformer(vocab_size=vocab_size,allow_halting=True)
    x = torch.randint(low=0,high=vocab_size-1,size=(2,50))
    true_y = torch.randint(low=0,high=vocab_size-1,size=(2,5))
    tgt_mask = make_tgt_mask(5)
    
    pred_y, p, h = model(x,true_y,tgt_mask=tgt_mask)
    print(pred_y.shape)# (max,step, batch_size, seq_len, dim)
    print(p.shape)# (max_step, batch_size)
    print(h.shape)# (batch_size), ループ回数
    print(h)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明') 
    parser.add_argument('--arg1', help='この引数の説明') # 必須の引数を追加
    args = parser.parse_args()
    main()
