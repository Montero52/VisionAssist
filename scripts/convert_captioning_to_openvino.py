import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import openvino as ov
import logging
import gc
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Internal imports
from src.main.model.model import ViT_Transformer
from config import vit_cfg, trans_cfg

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OpenVINO-Converter")

def manual_attn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    """Thay thế hàm SDPA lỗi bằng logic Attention thủ công để OpenVINO có thể 'đọc' được."""
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / (query.size(-1) ** 0.5) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=False)
    return attn_weight @ value

F.scaled_dot_product_attention = manual_attn

class VisionEncoderWrapper(nn.Module):
    """
    Wraps the ViT Encoder and Projection layer to export as a single OpenVINO component.
    """
    def __init__(self, original_model: ViT_Transformer):
        super().__init__()
        self.encoder = original_model.encoder
        self.proj = original_model.proj

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Input images: [1, 3, 224, 224]
        features = self.encoder(images)
        # Output project to decoder dimension: [1, 197, 768]
        encoder_out = self.proj(features)
        return encoder_out

class TextDecoderWrapper(nn.Module):
    """
    Wraps the Transformer Decoder for autoregressive text generation.
    """
    def __init__(self, original_model: ViT_Transformer):
        super().__init__()
        self.decoder = original_model.decoder

    def forward(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [1, 32]
            encoder_hidden_states: [1, 197, 768]
            attention_mask: [1, 1, 32, 32] (Pre-calculated causal + padding mask)
        """
        return self.decoder(input_ids, encoder_hidden_states, attention_mask)

def convert_captioning_model():
    # 1. Setup paths
    ir_base_path = Path("checkpoints/ir/captioning")
    encoder_path = ir_base_path / "encoder" / "encoder.xml"
    decoder_path = ir_base_path / "decoder" / "decoder.xml"
    
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_path.parent.mkdir(parents=True, exist_ok=True)

# ==========================================================
    # 2. LOAD WEIGHTS & DYNAMIC MODEL INITIALIZATION
    # ==========================================================
    WEIGHTS_PATH = "checkpoints/vizwiz_adapted_final.pth"
    
    if not os.path.exists(WEIGHTS_PATH):
        logger.error(f"Weights file not found at: {WEIGHTS_PATH}")
        return

    # Load checkpoint to CPU to save memory
    logger.info(f"Loading VizWiz weights from: {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    
    # Extract state_dict if it's a full training checkpoint
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # MLOps Optimization: Detect actual vocab_size from weights to prevent mismatch
    # This ensures compatibility regardless of tokenizer pruning
    actual_vocab_size = state_dict['decoder.token_embed.weight'].shape[0]
    logger.info(f"Detected actual vocab_size from weights: {actual_vocab_size}")
    trans_cfg['vocab_size'] = actual_vocab_size

    # Initialize model once with the synchronized configuration
    logger.info("Initializing ViT_Transformer with dynamic vocab_size...")
    full_model = ViT_Transformer(vit_config=vit_cfg, trans_cfg=trans_cfg)
    
    # Load weights into the architecture
    full_model.load_state_dict(state_dict) 
    full_model.eval()
    logger.info("Weights loaded successfully! Model is synchronized and ready for export.")

    # ==========================================
    # ENCODER CONVERSION
    # ==========================================
    logger.info("Starting Encoder conversion (Static Shape [1, 3, 224, 224])...")
    encoder_wrapper = VisionEncoderWrapper(full_model)
    
    dummy_image = torch.randn(1, 3, 224, 224)
    
    try:
        ov_encoder = ov.convert_model(encoder_wrapper, example_input=dummy_image)
        ov.save_model(ov_encoder, str(encoder_path), compress_to_fp16=True)
        logger.info(f"Encoder IR saved successfully at: {encoder_path}")
    except Exception as e:
        logger.error(f"Failed to convert Encoder: {e}")
        return

    # Clean up RAM immediately after Encoder
    logger.info("Cleaning up RAM after Encoder conversion...")
    del encoder_wrapper
    del ov_encoder
    gc.collect()

    # ==========================================
    # DECODER CONVERSION
    # ==========================================
    logger.info("Starting Decoder conversion (Static Shape, Max Sequence Length = 32)...")
    decoder_wrapper = TextDecoderWrapper(full_model)
    
    # Define Static Shapes for Decoder
    MAX_SEQ_LEN = 32
    HIDDEN_DIM = trans_cfg.get("dim", 768)
    NUM_PATCHES = 197 # (224/16)^2 + 1 class token
    
    dummy_input_ids = torch.zeros((1, MAX_SEQ_LEN), dtype=torch.long)
    dummy_encoder_out = torch.randn(1, NUM_PATCHES, HIDDEN_DIM)
    dummy_mask = torch.randn(1, 1, MAX_SEQ_LEN, MAX_SEQ_LEN) # Causal + Padding mask

    try:
        ov_decoder = ov.convert_model(
            decoder_wrapper, 
            example_input=(dummy_input_ids, dummy_encoder_out, dummy_mask)
        )
        ov.save_model(ov_decoder, str(decoder_path), compress_to_fp16=True)
        logger.info(f"Decoder IR saved successfully at: {decoder_path}")
    except Exception as e:
        logger.error(f"Failed to convert Decoder: {e}")
        return

    # Final Cleanup
    logger.info("Finalizing conversion and cleaning up resources...")
    del decoder_wrapper
    del full_model
    gc.collect()
    
    logger.info("=== STEP 3 COMPLETED: CAPTIONING MODELS CONVERTED TO OPENVINO IR ===")

if __name__ == "__main__":
    convert_captioning_model()
