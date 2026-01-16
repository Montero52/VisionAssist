from torch import nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)