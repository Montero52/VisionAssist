import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, T, D = query.shape
        H = self.num_heads

        # 1. Chiếu Q, K, V và reshape cho multi-head attention
        q = self.q_proj(query).view(B, T, H, self.head_dim).transpose(1, 2) # [B, H, T, D_h]
        k = self.k_proj(key).view(B, -1, H, self.head_dim).transpose(1, 2)   # [B, H, S, D_h]
        v = self.v_proj(value).view(B, -1, H, self.head_dim).transpose(1, 2) # [B, H, S, D_h]

        # 2. Tính Attention Scores
        # Scores: [B, H, T, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 3. Áp dụng Mask (Nếu có)
        if mask is not None:
            # Mask có thể là [B, 1, 1, S] (padding) hoặc [1, 1, T, T] (look-ahead)
            if mask.ndim == 4:
                scores = scores + mask  # Thêm mask (giá trị -inf)
            else: # Dành cho padding mask [B, 1, S]
                scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        # 4. Tính Weights và áp dụng Dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # 5. Tính đầu ra, ghép các đầu và chiếu lại
        # Output: [B, H, T, D_h] -> [B, T, H, D_h] -> [B, T, D]
        output = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(output)