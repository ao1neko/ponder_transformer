import argparse
from vanilla_transformer import Transformer

import torch
import torch.nn as nn


class PonderTransformer(nn.Module):
    def __init__(
        self,vocab_size ,emb_dim=300, max_steps=20, allow_halting=False,num_tokens = 10
    ):
        """ PonderNet + TransformerBlock

        Args:
            vocab_size (int): 
            emb_dim (int, optional): Defaults to 300.
            max_steps (int, optional): Defaults to 20.
            allow_halting (bool, optional): If True, then the forward pass is allowed to halt before
                                            reaching the maximum steps. Defaults to False.
            num_tokens (int, optional): number of output types. Defaults to 10.
        """
        super().__init__()
        self.max_steps = max_steps
        self.emb_dim = emb_dim
        self.allow_halting = allow_halting
        self.num_tokens = num_tokens
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.transformer_layer = Transformer(emb_dim=emb_dim)
        self.output_layer = nn.Linear(emb_dim, num_tokens)
        self.lambda_layer = nn.Linear(emb_dim, 1)

        
    def forward(self, x):
        mask = (x == 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x)
        
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
            if n == self.max_steps:
                lambda_n = x.new_ones(batch_size)  # (batch_size,)
            else:
                
                lambda_n = torch.sigmoid(self.lambda_layer(x[:,0]))[
                    :, 0
                ]  # (batch_size,)
            
            # Store releavant outputs
            y_list.append(self.output_layer(x))  # (batch_size,)
            p_list.append(un_halted_prob * lambda_n)  # (batch_size,)

            halting_step = torch.maximum(
                n
                * (halting_step == 0)
                * torch.bernoulli(lambda_n).to(torch.long),
                halting_step,
            )

            # Prepare for next iteration
            un_halted_prob = un_halted_prob * (1 - lambda_n)
            x = self.transformer_layer(x,mask)

            # Potentially stop if all samples halted
            if self.allow_halting and (halting_step > 0).sum() == batch_size:
                break

        y = torch.stack(y_list)
        p = torch.stack(p_list)

        return y, p, halting_step


def main():
    vocab_size = 500
    model = PonderTransformer(vocab_size=vocab_size)
    x = torch.randint(low=0,high=vocab_size-1,size=(2,50))
    pred_y, p, h = model(x)    
    print(pred_y.shape)# (max,step, batch_size, seq_len, dim)
    print(p.shape)# (max_step, batch_size)
    print(h.shape)# (batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明') 
    parser.add_argument('--arg1', help='この引数の説明') # 必須の引数を追加
    args = parser.parse_args()
    main()
