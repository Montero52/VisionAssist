import torch
from torch import nn
# Đảm bảo đường dẫn import này đúng với cấu trúc thư mục của bạn
# Nếu file transformer.py nằm cùng thư mục vit.py thì dùng dấu chấm (.)
from .vit_layers import TransformerEncoderBlock 

class ViT(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=32,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image_size phải chia hết cho patch_size"

        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        # Lớp Linear để chiếu patch thành vector embedding
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        # Token đặc biệt [CLS] và Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Các khối Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)

        # --- SỬA 1: Bỏ hoàn toàn self.head vì không dùng Classification ---
        # self.head = nn.Linear(embed_dim, num_classes)

        # Gọi hàm khởi tạo trọng số
        self._init_weights()

    def _init_weights(self):
        # 1. Khởi tạo pos_embed và cls_token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # --- SỬA 2: Áp dụng hàm _init_vit_weights cho toàn bộ mạng con ---
        self.apply(self._init_vit_weights)

    def _init_vit_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
                
    def forward(self, x):
        B, C, H, W = x.shape
        
        # --- SỬA 3: Dùng đúng tên biến self.patch_size ---
        P = self.patch_size 

        # Xử lý cắt ảnh thành patches (Thủ công)
        # Input: [B, C, H, W] -> Output: [B, Num_Patches, Patch_Dim]
        patches = x.unfold(2, P, P).unfold(3, P, P)
        patches = patches.contiguous().view(B, C, -1, P, P)
        patches = patches.permute(0, 2, 1, 3, 4) 
        patches = patches.flatten(2)  

        # Embedding
        x = self.patch_embed(patches)

        # Thêm [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Thêm Positional Embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Đi qua các khối Transformer
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # --- SỬA 4: Trả về toàn bộ chuỗi features (cho Image Captioning) ---
        # Bỏ đoạn tách cls_output và head
        # cls_output = x[:, 0]  
        # logits = self.head(cls_output)
        
        return x  # Shape: [Batch, Num_Patches + 1, Embed_Dim]