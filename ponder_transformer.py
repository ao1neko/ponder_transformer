import argparse

import torch
import torch.nn as nn
from util import make_tgt_mask
from einops import rearrange, repeat

class PonderTransformer(nn.Module):
    def __init__(
        self,vocab_size ,emb_dim=300, max_steps=20, allow_halting=False,nhead=10,num_token=100
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
        self.num_token = num_token
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.output_layer = nn.Linear(emb_dim, self.num_token)
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
                
                lambda_n = torch.sigmoid(self.lambda_layer(x[:,0]))[:, 0]  # (batch_size,)
            
            # Store releavant outputs
            #y_list.append(self.output_layer(x))  # (batch_size,)
            h = true_y.clone()
            for _ in range(n):
                h = self.transformer_decoder_layer(tgt=h,memory=x,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask)
            h = self.output_layer(h)
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


class ReconstructionLoss(nn.Module):
    """Weighted average of per step losses.

    Parameters
    ----------
    loss_func : callable
        Loss function that accepts `y_pred` and `y_true` as arguments. Both
        of these tensors have shape `(batch_size,)`. It outputs a loss for
        each sample in the batch.
    """

    def __init__(self, loss_func):
        super().__init__()

        self.loss_func = loss_func

    def forward(self, p, y_pred, y_true):
        """Compute loss.

        Parameters
        ----------
        p : torch.Tensor
            Probability of halting of shape `(max_steps, batch_size)`.

        y_pred : torch.Tensor
            Predicted outputs of shape `(max_steps, batch_size)`.

        y_true : torch.Tensor
            True targets of shape `(batch_size,)`.

        Returns
        -------
        loss : torch.Tensor
            Scalar representing the reconstruction loss. It is nothing else
            than a weighted sum of per step losses.
        """
        max_steps, _ = p.shape
        total_loss = p.new_tensor(0.0)

        for n in range(max_steps):
            seq_len = y_true.shape[1]
            arranged_y_true = rearrange(y_true.clone(), 'b s -> (b s)')
            arranged_y_pred = rearrange(y_pred[n].clone(), 'b s d-> (b s) d')
            loss_per_sample = p[n] * self.loss_func(arranged_y_pred,arranged_y_true) * seq_len  # (batch_size,)
            total_loss = total_loss + loss_per_sample.mean()  # (1,)

        return total_loss


class RegularizationLoss(nn.Module):
    """Enforce halting distribution to ressemble the geometric distribution.

    Parameters
    ----------
    lambda_p : float
        The single parameter determining uniquely the geometric distribution.
        Note that the expected value of this distribution is going to be
        `1 / lambda_p`.

    max_steps : int
        Maximum number of pondering steps.
    """

    def __init__(self, lambda_p, max_steps=20):
        super().__init__()

        p_g = torch.zeros((max_steps,))
        not_halted = 1.0

        for k in range(max_steps):
            p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)

        self.register_buffer("p_g", p_g)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, p):
        """Compute loss.

        Parameters
        ----------
        p : torch.Tensor
            Probability of halting of shape `(steps, batch_size)`.

        Returns
        -------
        loss : torch.Tensor
            Scalar representing the regularization loss.
        """
        steps, batch_size = p.shape

        p = p.transpose(0, 1)  # (batch_size, max_steps)
        p_g_batch = self.p_g[None, :steps].expand_as(
            p
        )  # (batch_size, max_steps)
        return self.kl_div(p, p_g_batch)



def main():
    vocab_size = 10
    model = PonderTransformer(vocab_size=vocab_size,allow_halting=True)
    x = torch.randint(low=0,high=vocab_size-1,size=(2,50))
    true_y = torch.randint(low=0,high=vocab_size-1,size=(2,5))
    tgt_mask = make_tgt_mask(5)
    
    pred_y, p, h = model(x,true_y,tgt_mask=tgt_mask)
    print(pred_y.shape)# (max,step, batch_size, seq_len, dim)
    print(p.shape)# (max_step, batch_size)
    print(h.shape)# (batch_size), ループ回数
    print(h)
    
    loss_rec_inst = ReconstructionLoss(nn.CrossEntropyLoss())
    loss_reg_inst = RegularizationLoss(lambda_p=0.1,max_steps=20)
    loss_rec = loss_rec_inst(p,pred_y,true_y,)
    loss_reg = loss_reg_inst(p,)
    
    loss_overall = loss_rec + 0.1 * loss_reg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明') 
    parser.add_argument('--arg1', help='この引数の説明') # 必須の引数を追加
    args = parser.parse_args()
    main()
