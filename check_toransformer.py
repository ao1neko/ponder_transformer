import torch
import torch.nn as nn
from util import make_tgt_mask, PositionalEncoder
from einops import rearrange, repeat



class CheckTransformerGenerater(nn.Module):
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
        
        y_list = []
        for _ in range(self.num_layers):
            x = self.transformer_encoder(x,src_key_padding_mask=src_key_padding_mask)
            y = self.output_layer(x[:,0])
            y_list.append(y)
            
        y_list = torch.stack(y_list)
        return y_list
    
    

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
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true, pad_id=0):
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
        batch_size  = y_pred.shape(1)
        total_loss = y_pred.new_tensor(0.0)
        
        y_pred = rearrange(y_pred, 'n b d-> b n d')
        loss_w = 
        
        for y_pred_n,y_true_token in zip(y_pred,y_true):
            sample_loss = y_pred.new_tensor(0.0)
            
            
            for i, (p_item,y_pred_token) in enumerate(zip(p_n,y_pred_n)):
                cross_loss  = self.loss_func(torch.unsqueeze(y_pred_token,dim=0),torch.unsqueeze(y_true_token,dim=0))
                sample_loss = sample_loss + p_item * cross_loss
            total_loss = total_loss + sample_loss
        return total_loss/batch_size
