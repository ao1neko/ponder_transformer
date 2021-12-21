import argparse

import torch
import torch.nn as nn
from util import make_tgt_mask, PositionalEncoder


class Transformer(nn.Module):
    def __init__(
        self,vocab_size ,emb_dim=512,num_layers=1,nhead=8,num_token=100,liner_dim=128
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
        self.liner_dim = liner_dim
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.pos_encoder = PositionalEncoder(d_model=emb_dim)
        self.transformer = nn.Transformer(d_model=emb_dim,nhead=nhead,num_encoder_layers=num_layers,num_decoder_layers=1,batch_first=True)
        self.output_layer = nn.Sequential(
            nn.Linear(emb_dim, self.num_token),
        )

        
    def forward(self, x:torch.Tensor,true_y:torch.Tensor,tgt_mask:torch.Tensor,src_key_padding_mask = None,tgt_key_padding_mask=None):
        x = self.embedding(x)
        true_y = self.embedding(true_y)   
        x = self.pos_encoder(x)
        true_y = self.pos_encoder(true_y)
        y = self.transformer(x,true_y,tgt_mask=tgt_mask,src_key_padding_mask=src_key_padding_mask,tgt_key_padding_mask=tgt_key_padding_mask)
        y = self.output_layer(y)
        return y

class TransformerrClassifier(nn.Module):
    def __init__(
        self,vocab_size ,emb_dim=512,num_layers=1,nhead=8,num_token=1
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
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.pos_encoder = PositionalEncoder(d_model=emb_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_token),
        )
        
    def forward(self, x:torch.Tensor,src_key_padding_mask = None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        y = self.transformer_encoder(x,src_key_padding_mask=src_key_padding_mask)
        y = self.output_layer(y[:,0])
        return y
    
class TransformerGenerater(nn.Module):
    def __init__(
        self,vocab_size ,emb_dim=128,num_layers=1,nhead=8,num_token=100,liner_dim=128
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
        self.liner_dim = liner_dim
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim)
        self.pos_encoder = PositionalEncoder(d_model=emb_dim)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim,nhead=nhead,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Sequential(
            nn.Linear(emb_dim, liner_dim),
            nn.ReLU(),
            nn.Linear(liner_dim, self.num_token)
        )

        
    def forward(self, x:torch.Tensor,src_key_padding_mask = None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        y = self.transformer_encoder(x,src_key_padding_mask=src_key_padding_mask)
        y = self.output_layer(y[:,0])
        return y
    
    
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
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self,y_pred, y_true, pad_id=0):
        """Compute loss.

        Parameters
        ----------
        p : torch.Tensor
            Probability of halting of shape `(max_steps, batch_size)`.

        y_pred : torch.Tensor
            Predicted outputs of shape `(max,step, batch_size ,seq_len, dim)`.

        y_true : torch.Tensor
            True targets of shape `(batch_size, seq_len)`.

        Returns
        -------
        loss : torch.Tensor
            Scalar representing the reconstruction loss. It is nothing else
            than a weighted sum of per step losses.
        """
        batch_size ,_ = y_true.shape
        total_loss = y_true.new_tensor(0.0)

        for y_pred_sequence,y_true_sequence in zip(y_pred,y_true):
            y_true_sequence = y_true_sequence[1:].clone()
            y_pred_sequence  = y_pred_sequence[:-1].clone()
            loss = y_true.new_tensor(0.0)
            count = y_true.new_tensor(0.0)
            
            for y_pred_token,y_true_token in zip(y_pred_sequence,y_true_sequence):
                if y_true_token.item() != pad_id:
                    count +=y_true.new_tensor(1.0)
                    loss = loss + self.loss_func(torch.unsqueeze(y_pred_token,dim=0),torch.unsqueeze(y_true_token,dim=0))
            total_loss = total_loss + loss/count 
        return total_loss/batch_size


   
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

    def forward(self,y_pred, y_true, pad_id=0):
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
        batch_size ,_ = y_true.shape
        total_loss = y_true.new_tensor(0.0)
        
        for y_pred_token,y_true_token in zip(y_pred,y_true):
            y_pred_token  = nn.functional.softmax(y_pred_token,dim=-1)
            total_loss = total_loss - torch.log(y_pred_token[y_true_token-1])
        return total_loss/batch_size


def main():
    vocab_size = 500
    model = Transformer(vocab_size=vocab_size,num_layers=1)
    x = torch.randint(low=0,high=vocab_size-1,size=(2,50))
    true_y = torch.randint(low=0,high=vocab_size-1,size=(2,5))
    tgt_mask = make_tgt_mask(5)
    pred_y = model(x,true_y,tgt_mask=tgt_mask)
    print(pred_y.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明') 
    parser.add_argument('--arg1', help='この引数の説明') # 必須の引数を追加
    args = parser.parse_args()
    main()

