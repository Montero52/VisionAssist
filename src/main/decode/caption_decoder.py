from torch import nn
import torch
from .t5_block import T5Block
import logging

logger = logging.getLogger("TransformerDecoder")

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int, 
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 32
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

    def forward(self, input_ids: torch.Tensor, encoder_out: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            input_ids: [Batch, Seq_Len] tensor (with Start Token)
            encoder_out: Encoded features from ViT [Batch, Num_Patches+1, Dim]
            mask: Combined causal and padding mask [1, 1, T, T] or equivalent for padding
        """
        B, T = input_ids.shape
        
        # Generate positional indices
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)

        # Truncate sequence if longer than max_len defined during init
        if T > self.max_len:
            logger.warning(f"Input sequence length ({T}) exceeded max_len ({self.max_len}). Truncating.")
            pos = pos[:self.max_len]
            input_ids = input_ids[:, :self.max_len]
            if mask is not None and mask.ndim == 4: 
                mask = mask[:, :, :self.max_len, :self.max_len]
        
        # Embed tokens and add positional encoding
        x = self.token_embed(input_ids) + self.pos_embed(pos)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, encoder_out, mask)

        x = self.norm(x)
        logits = self.fc_out(x)
        return logits