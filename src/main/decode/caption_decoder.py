from torch import nn
import torch
from .t5_block import T5Block

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size, 
        dim= 512,
        num_heads = 8,
        num_layers = 6,
        ff_dim = 2048,
        dropout = 0.1,
        max_len = 32
    ):
        
        super().__init__()
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)

        self.layers = nn.ModuleList([
            T5Block(dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, encoder_out, mask=None):
        B, T = input_ids.shape
        
        # Tạo pos indices
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)

        if T > self.max_len:
            pos = pos[:self.max_len] #pos = pos[:, :self.max_len]
            input_ids = input_ids[:, :self.max_len]
            if mask is not None: mask = mask[:, :self.max_len] # Cập nhật mask tương ứng

        x = self.token_embed(input_ids) + self.pos_embed(pos)

        for layer in self.layers:
            x = layer(x, encoder_out, mask)

        x = self.norm(x)
        logits = self.fc_out(x)
        return logits