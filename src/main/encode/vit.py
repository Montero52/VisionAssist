import torch
import timm
from torch import nn
from .vit_layers import TransformerEncoderBlock 

class ViT(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,       # ViT chuẩn thường dùng patch 16 hoặc 32
        in_channels=3,
        embed_dim=768,       # Quan trọng: Dùng để chọn model
        depth=12,            # (Không dùng nữa vì load từ pretrained)
        num_heads=12,        # (Không dùng nữa)
        mlp_ratio=4.0,       # (Không dùng nữa)
        dropout=0.0,         # Dropout nên để thấp hoặc 0 khi finetune
        pretrained=True      # Cờ để bật chế độ tải trọng số Google
    ):
        super().__init__()
        
        # 1. Tự động chọn phiên bản ViT dựa trên embed_dim
        # Google đặt tên model theo quy tắc: vit_<size>_patch<size>_<res>
        
        model_name = ""
        
        if embed_dim == 768:
            # ViT-Base (Chuẩn nhất)
            model_name = "vit_base_patch16_224"
        elif embed_dim == 384:
            # ViT-Small (Nhẹ hơn)
            model_name = "vit_small_patch16_224"
        elif embed_dim == 1024:
            # ViT-Large (Rất nặng)
            model_name = "vit_large_patch16_224"
        else:
            # Mặc định an toàn nếu config lạ
            print(f"Cảnh báo: embed_dim={embed_dim} không khớp chuẩn ViT. Đang dùng mặc định ViT-Base.")
            model_name = "vit_base_patch16_224"
            
        print(f"Đang khởi tạo Pretrained ViT: {model_name} (Pretrained={pretrained})")
        
        # 2. Load model từ thư viện timm
        # drop_rate: Dropout cho các lớp Linear
        # attn_drop_rate: Dropout cho Attention
        self.vit = timm.create_model(
            model_name, 
            pretrained=pretrained,
            drop_rate=dropout,
            attn_drop_rate=dropout
        )
        
        # 3. [QUAN TRỌNG] Loại bỏ lớp phân loại (Classification Head)
        # Vì ta làm Image Captioning, ta cần đặc trưng (features), 
        # không cần model đoán đây là con chó hay con mèo.
        self.vit.head = nn.Identity()
        
        # Lưu lại embed_dim thực tế để kiểm tra nếu cần
        self.embed_dim = self.vit.embed_dim

    def forward(self, x):
        """
        Input: [Batch, 3, 224, 224]
        Output: [Batch, Num_Patches + 1, Embed_Dim] 
        (Ví dụ: [Batch, 197, 768])
        """
        # forward_features: Hàm chuyên dụng của timm để lấy vector đặc trưng
        # Nó tự động xử lý Patch Embedding, CLS token, Positional Embedding
        features = self.vit.forward_features(x)
        
        return features