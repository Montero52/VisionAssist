from torch import nn
from ..modules.attention import Attention
from ..modules.feedforward import FFN

class T5Block(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # 1. Self-Attention (Autoregressive)
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = Attention(dim, num_heads, dropout)

        # 2. Cross-Attention (Encoder-Decoder Attention)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = Attention(dim, num_heads, dropout)

        # 3. Feed-Forward
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, ff_dim, dropout)

    def forward(self, x, encoder_out, mask=None):
        # Self-Attention (mask=mask, thường là look-ahead mask)
        x = x + self.self_attn(self.norm1(x), x, x, mask=mask)

        # Cross-Attention (key/value từ encoder_out)
        x = x + self.cross_attn(self.norm2(x), encoder_out, encoder_out)

        # Feed-Forward
        x = x + self.ffn(self.norm3(x))
        return x