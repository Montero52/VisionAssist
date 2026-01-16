from torch import nn
from src.main.modules.attention import Attention
from src.main.modules.feedforward import FFN

class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout=0.1):
        super().__init__()
        ff_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, ff_dim, dropout)

    def forward(self, x):
        # Self-Attention
        x = x + self.attn(self.norm1(x), x, x)
        # Feed-Forward
        x = x + self.ffn(self.norm2(x))
        return x