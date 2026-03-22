import torch
import timm
from torch import nn
import logging

logger = logging.getLogger("ViT-Encoder")

class ViT(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,       # Standard ViT often uses patch 16 or 32
        in_channels=3,
        embed_dim=768,       # Used to select the model version
        depth=12,            # (Not used, loaded from pretrained)
        num_heads=12,        # (Not used)
        mlp_ratio=4.0,       # (Not used)
        dropout=0.0,         # Low dropout for finetuning
        pretrained=True      # Load Google weights
    ):
        super().__init__()
        
        # 1. Automatically select ViT version based on embed_dim
        # Google model naming: vit_<size>_patch<size>_<res>
        
        model_name = ""
        
        if embed_dim == 768:
            # ViT-Base (Standard)
            model_name = "vit_base_patch16_224"
        elif embed_dim == 384:
            # ViT-Small (Lightweight)
            model_name = "vit_small_patch16_224"
        elif embed_dim == 1024:
            # ViT-Large (Heavy)
            model_name = "vit_large_patch16_224"
        else:
            # Safe default for unknown configs
            logger.warning(f"embed_dim={embed_dim} does not match ViT standards. Using default ViT-Base.")
            model_name = "vit_base_patch16_224"
            
        logger.info(f"Initializing Pretrained ViT: {model_name} (Pretrained={pretrained})")
        
        # 2. Load model from timm library
        # drop_rate: Linear layer dropout
        # attn_drop_rate: Attention dropout
        self.vit = timm.create_model(
            model_name, 
            pretrained=pretrained,
            drop_rate=dropout,
            attn_drop_rate=dropout
        )
        
        # 3. [IMPORTANT] Remove Classification Head
        # For Image Captioning, we need features, not class predictions.
        self.vit.head = nn.Identity()
        
        self.embed_dim = self.vit.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: [Batch, 3, 224, 224]
        Output: [Batch, Num_Patches + 1, Embed_Dim] 
        (Example: [Batch, 197, 768])
        """
        # forward_features: Specialized timm function to extract feature vectors
        # Handles Patch Embedding, CLS token, and Positional Embedding
        features = self.vit.forward_features(x)
        
        return features
