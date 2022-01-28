import argparse

import torch
import torch.nn as nn
from util import make_tgt_mask, PositionalEncoder,RandomPositionalEncoder
from einops import rearrange, repeat
import math
import matplotlib.pyplot as plt

class PonderTransformer(nn.Module):
    def __init__(
        self,vocab_size ,emb_dim=512, max_steps=20, allow_halting=False,nhead=8,num_token=100,liner_dim=128
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
        self.rand_pos_encoder = RandomPositionalEncoder(d_model=emb_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.norm = nn.LayerNorm(emb_dim,eps = 1e-04)
        
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1,norm=self.norm)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=1,norm=self.norm)
        self.output_layer = nn.Linear(emb_dim, self.num_token)
        self.lambda_layer = nn.Linear(emb_dim, 1)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        self.lambda_layer.bias.data.zero_()
        self.lambda_layer.weight.data.uniform_(-initrange, initrange)
        
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
        x = self.rand_pos_encoder(x)
        
        true_y = self.embedding(true_y)
        true_y = self.pos_encoder(true_y)

        
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
            x = self.transformer_encoder(x,src_key_padding_mask=src_key_padding_mask)
            
            if n == self.max_steps:
                lambda_n = x.new_ones(batch_size)  # (batch_size,)
            else:
                lambda_n = torch.sigmoid(self.lambda_layer(x[:,0]))[:, 0]  # (batch_size,)
            
            # Store releavant outputs
            #y_list.append(self.output_layer(x))  # (batch_size,)
            h = true_y.clone()
            for _ in range(n):
                h = self.transformer_decoder(tgt=h,memory=x,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=src_key_padding_mask)
            h = self.output_layer(h)
            y_list.append(h)
            
            
            
            p_list.append(un_halted_prob * lambda_n)  # (batch_size,)
            
            p_tensor = torch.stack(p_list)
            if self.training:
                halting_step = torch.maximum(n*(halting_step == 0)*torch.bernoulli(lambda_n).to(torch.long),halting_step,)
            else:
                halting_step = torch.maximum(n*(halting_step == 0)*(p_tensor.sum(dim=0) > 0.5).to(torch.long),halting_step,)
            
            # Prepare for next iteration
            un_halted_prob = un_halted_prob * (1 - lambda_n)

            # Potentially stop if all samples halted
            if self.allow_halting and (halting_step > 0).sum() == batch_size:
                break
        
        y = torch.stack(y_list)
        p = torch.stack(p_list)

        return y, p, halting_step


class PonderTransformerClassifier(nn.Module):
    def __init__(
        self,vocab_size ,emb_dim=512, max_steps=20, allow_halting=False,nhead=8,num_token=1
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
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.pos_encoder = PositionalEncoder(d_model=emb_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        self.output_layer = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_token)
        )
        self.lambda_layer = nn.Linear(emb_dim, 1)

        
    def forward(self, x:torch.Tensor,src_key_padding_mask = None):
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
            x = self.transformer_encoder(x,src_key_padding_mask=src_key_padding_mask)
            if n == self.max_steps:
                lambda_n = x.new_ones(batch_size)  # (batch_size,)
            else:
                
                lambda_n = torch.sigmoid(self.lambda_layer(x[:,0]))[:, 0]  # (batch_size,)
            
            # Store releavant outputs
            #y_list.append(self.output_layer(x))  # (batch_size,)
            h = self.output_layer(x[:,0])
            y_list.append(h)
            
            
            
            p_list.append(un_halted_prob * lambda_n)  # (batch_size,)

            halting_step = torch.maximum(
                n * (halting_step == 0)* torch.bernoulli(lambda_n).to(torch.long),
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




class PonderTransformerGenerater(nn.Module):
    def __init__(
        self,vocab_size ,emb_dim=128, max_steps=20, allow_halting=False,nhead=8,num_token=100,liner_dim=128
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
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim,padding_idx=0)
        self.pos_encoder = PositionalEncoder(d_model=emb_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=1)
        self.lambda_layer = nn.Linear(emb_dim, 1)
        self.output_layer = nn.Linear(emb_dim,self.num_token)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        self.lambda_layer.bias.data.zero_()
        self.lambda_layer.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, x:torch.Tensor,src_key_padding_mask = None):
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
            x = self.transformer_encoder(x,src_key_padding_mask=src_key_padding_mask)
            if n == self.max_steps:
                lambda_n = x.new_ones(batch_size)  # (batch_size,)
            else:
                lambda_n = torch.sigmoid(self.lambda_layer(x[:,0]))[:, 0]  # (batch_size,)
            
            # Store releavant outputs
            #y_list.append(self.output_layer(x))  # (batch_size,)
            h = self.output_layer(x[:,0])
            y_list.append(h)
            
            
            
            p_list.append(un_halted_prob * lambda_n)  # (batch_size,)
            
            halting_step = torch.maximum(n*(halting_step == 0)*torch.bernoulli(lambda_n).to(torch.long),halting_step,)

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

    def __init__(self):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    def forward(self, p, y_pred, y_true, pad_id=0):
        """Compute loss.

        Parameters
        ----------
        p : torch.Tensor
            Probability of halting of shape `(max_steps, batch_size)`.

        y_pred : torch.Tensor
            Predicted outputs of shape `(max,step, batch_size, seq_len, dim)`.

        y_true : torch.Tensor
            True targets of shape `(batch_size, seq_len)`.

        Returns
        -------
        loss : torch.Tensor
            Scalar representing the reconstruction loss. It is nothing else
            than a weighted sum of per step losses.
        """
        max_steps, batch_size, seq_len, _ = y_pred.shape
        y_pred = rearrange(y_pred, 'n b s d-> n d b s')
        y_pred_sequence= y_pred[:,:,:,:-1].clone()
        y_true_sequence = y_true[:,1:].clone()
        y_true_sequence = y_true_sequence.expand(max_steps,batch_size,seq_len-1)
        mask = (y_true_sequence != pad_id).float()   
        return (p * (self.loss_func(y_pred_sequence,y_true_sequence)*mask).sum(dim=-1)/torch.count_nonzero(mask,dim=-1)).sum(dim=0).mean()



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

    def __init__(self, lambda_p, device=torch.device('cpu'), max_steps=20):
        super().__init__()

        p_g = torch.zeros((max_steps,)).to(device)
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
    

class MyRegularizationLoss(nn.Module):
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

    def __init__(self, lambda_p, device=torch.device('cpu'), max_steps=20):
        super().__init__()

        self.p_g = torch.ones((max_steps,)).to(device)
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

        p = p.transpose(0, 1)  # (batch_size, max_steps)
        p_g_batch = self.p_g.expand_as(p)  # (batch_size, max_steps)
        return self.kl_div(torch.exp(p), p_g_batch)


class ClassifyingReconstructionLoss(nn.Module):
    """Weighted average of per step losses.
    Parameters
    ----------
    loss_func : callable
        Loss function that accepts `y_pred` and `y_true` as arguments. Both
        of these tensors have shape `(batch_size,)`. It outputs a loss for
        each sample in the batch.
    """

    def __init__(self):
        super().__init__()

    def forward(self, p, y_pred, y_true, pad_id=0):
        """Compute loss.
        Parameters
        ----------
        p : torch.Tensor
            Probability of halting of shape `(max_steps, batch_size)`.
        y_pred : torch.Tensor
            Predicted outputs of shape `(max,step, batch_size, dim)`.
        y_true : torch.Tensor
            True targets of shape `(batch_size)`.
        Returns
        -------
        loss : torch.Tensor
            Scalar representing the reconstruction loss. It is nothing else
            than a weighted sum of per step losses.
        """
        max_steps, batch_size = p.shape
        total_loss = p.new_tensor(0.0)
        
        p = rearrange(p, 'n b-> b n')
        y_pred = rearrange(y_pred, 'n b d-> b n d')
        
        for p_n,y_pred_n,y_true_token in zip(p,y_pred,y_true):
            sample_loss = p.new_tensor(0.0)
            for p_item,y_pred_token in zip(p_n,y_pred_n):
                y_pred_token  = nn.functional.softmax(y_pred_token,dim=-1)
                sample_loss = sample_loss - p_item * torch.log(y_pred_token[y_true_token-1])# yokunaikedo
                
            total_loss = total_loss + sample_loss
        return total_loss/batch_size

class GeneratingReconstructionLoss(nn.Module):
    """Weighted average of per step losses.

    Parameters
    ----------
    loss_func : callable
        Loss function that accepts `y_pred` and `y_true` as arguments. Both
        of these tensors have shape `(batch_size,)`. It outputs a loss for
        each sample in the batch.
    """

    def __init__(self):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

    def forward(self, p, y_pred, y_true, pad_id=0):
        """Compute loss.

        Parameters
        ----------
        p : torch.Tensor
            Probability of halting of shape `(max_steps, batch_size)`.

        y_pred : torch.Tensor
            Predicted outputs of shape `(max,step, batch_size, seq_len, dim)`.

        y_true : torch.Tensor
            True targets of shape `(batch_size, seq_len)`.

        Returns
        -------
        loss : torch.Tensor
            Scalar representing the reconstruction loss. It is nothing else
            than a weighted sum of per step losses.
        """
        max_steps, batch_size = p.shape
        y_true_sequence = y_true.expand(max_steps, batch_size).clone()
        y_pred = rearrange(y_pred, 'n b d-> n d b')
        return (p * self.loss_func(y_pred,y_true_sequence)).sum(dim=0).mean()

def main():
    torch.manual_seed(6)
    torch.cuda.manual_seed_all(6)
    torch.backends.cudnn.deterministic = True
    vocab_size = 10
    model = PonderTransformer(vocab_size=vocab_size,allow_halting=False)
    x = torch.randint(low=0,high=vocab_size-1,size=(2,50))
    true_y = torch.randint(low=0,high=vocab_size-1,size=(2,5))
    tgt_mask = make_tgt_mask(5)
    
    pred_y, p, h = model(x,true_y,tgt_mask=tgt_mask)
    print(pred_y.shape)# (max,step, batch_size, seq_len, dim)
    print(p.shape)# (max_step, batch_size)
    print(h.shape)# (batch_size), ループ回数
    print(h)
    
    loss_rec_inst = ReconstructionLoss()
    loss_reg_inst = RegularizationLoss(lambda_p=1/20,max_steps=20)
    #loss_rec = loss_rec_inst(p,pred_y,true_y,)
    p = torch.Tensor([[5.9215e-01, 5.9208e-01],[2.6437e-01, 2.5560e-01],[9.8978e-02, 9.6060e-02],
        [3.1605e-02, 3.6037e-02],
        [9.1340e-03, 1.2945e-02],
        [2.6090e-03, 4.5568e-03],
        [7.5887e-04, 1.7224e-03],
        [2.5966e-04, 6.1499e-04],
        [8.1615e-05, 2.4406e-04],
        [3.1655e-05, 9.0201e-05],
        [1.0462e-05, 3.5348e-05],
        [3.9676e-09, 1.3255e-05],
        [1.2760e-06, 4.6339e-06],
        [4.9438e-07, 1.7711e-06],
        [1.7994e-07, 5.1751e-07],
        [5.7837e-08, 1.4168e-07],
        [2.1115e-01, 4.7183e-08],
        [7.4858e-09, 1.4922e-08],
        [2.5357e-09, 4.5277e-09],
        [1.4573e-09, 2.0167e-09]])
    p = torch.softmax(p,dim=0)
    loss_reg = loss_reg_inst(p,)
    print(loss_reg)
    #loss_overall = loss_rec + 0.1 * loss_reg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明') 
    parser.add_argument('--arg1', help='この引数の説明') # 必須の引数を追加
    args = parser.parse_args()
    main()
