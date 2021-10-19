import argparse

import torch
import torch.nn as nn

from transformer.transformer import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, emb_dim=300, n_layers=1, attn_heads=12, dropout=0.1):
        """vanila transformer

        Args:
            vocab_size (int)
            emb_dim (int, optional): number of embedding dimmentions. Defaults to 300.
            n_layers (int, optional): number of transformer blocks. Defaults to 1.
            attn_heads (int, optional): Defaults to 12.
            dropout (float, optional): dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = emb_dim * 4
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(emb_dim, attn_heads, emb_dim * 4, dropout) for _ in range(n_layers)])

    def forward(self, x,mask):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x


def main():
    vocab_size = 500
    x = torch.randint(low=0, high=vocab_size, size=(2, 10))
    model = Transformer()
    print(model)
    pred_y = model(x)
    print(pred_y.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='このプログラムの説明')
    parser.add_argument('--arg1', help='この引数の説明')  # 必須の引数を追加
    args = parser.parse_args()
    main()
