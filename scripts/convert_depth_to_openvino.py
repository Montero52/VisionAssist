import os
import torch
import gc
import logging
import openvino as ov
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

# Configure logging according to professional standards
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration ---
MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
IR_OUTPUT_DIR = "checkpoints/ir/depth"
MODEL_XML_PATH = os.path.join(IR_OUTPUT_DIR, "depth_anything_v2_small.xml")

def convert_depth_model_to_ir() -> None:
    """
    Converts Depth-Anything-V2 to OpenVINO IR format with FP16 precision.
    Optimized for systems with limited RAM (8GB).
    """
    os.makedirs(IR_OUTPUT_DIR, exist_ok=True)
    
    logging.info(f"Loading PyTorch model: {MODEL_NAME}")
    try:
        # 1. Load PyTorch model and processor
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForDepthEstimation.from_pretrained(
            MODEL_NAME, 
            attn_implementation="eager" # Tắt SDPA để OpenVINO biên dịch dễ dàng
        )
        model.eval()

        # 2. Prepare Dummy Input (Important for static shape optimization)
        # Using the recommended size for Depth-Anything-V2-Small
        h, w = processor.size['height'], processor.size['width']
        example_input = torch.randn(1, 3, h, w)

        logging.info("Starting OpenVINO conversion (this may take a minute)...")
        
        # 3. Use the modern OpenVINO 2.0 API (direct conversion)
        # This replaces the old 'openvino.tools.mo'
        ov_model = ov.convert_model(model, example_input=example_input)

        # 4. Save to IR format with FP16 compression to save RAM and Disk
        # compress_to_fp16=True is the key for performance on CPU/iGPU
        ov.save_model(ov_model, MODEL_XML_PATH, compress_to_fp16=True)
        
        logging.info(f"Success! IR Model saved at: {MODEL_XML_PATH}")

        # 5. CRITICAL: Memory Management for 8GB RAM
        del model
        del ov_model
        gc.collect()
        logging.info("PyTorch model cleared from RAM.")

    except Exception as e:
        logging.error(f"Conversion failed: {e}")

if __name__ == "__main__":
    convert_depth_model_to_ir()