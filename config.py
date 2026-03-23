import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- SMART PATH CONFIGURATION ---
# 1. Prioritize temporary data on Colab (High speed)
if os.path.exists("/content/temp_data/flickr30k_images"):
    DATA_ROOT = "/content/temp_data/flickr30k_images"
    logger.info(f"Dataset Root: {DATA_ROOT}")
    
    # [IMPORTANT] Check where the images are located
    # Case 1: flickr30k_images/flickr30k_images/*.jpg (Common when unzipping)
    if os.path.exists(os.path.join(DATA_ROOT, "flickr30k_images")):
        image_dir = os.path.join(DATA_ROOT, "flickr30k_images")
    # Case 2: flickr30k_images/images/*.jpg (Old code structure)
    elif os.path.exists(os.path.join(DATA_ROOT, "images")):
        image_dir = os.path.join(DATA_ROOT, "images")
    # Case 3: Images are directly in DATA_ROOT
    else:
        image_dir = DATA_ROOT
        
    logger.info(f"Automatically detected image directory at: {image_dir}")

# 2. Running on Google Drive
elif os.path.exists("/content/drive/MyDrive/01_Dev_Projects/Video_Captioning/src/data/DatasetFlickr30k"):
    DATA_ROOT = "/content/drive/MyDrive/01_Dev_Projects/Video_Captioning/src/data/DatasetFlickr30k"
    image_dir = os.path.join(DATA_ROOT, "images") # Drive is manually managed, so this should be correct
    logger.info(f"Running on Drive: {image_dir}")

# 3. Running Locally
else:
    DATA_ROOT = os.path.join(BASE_DIR, "src", "data", "flickr8k")
    image_dir = os.path.join(DATA_ROOT, "images")
    logger.info(f"Running Locally: {image_dir}")

# Caption path configuration
if "temp_data" in DATA_ROOT:
    caption_dir = "/content/drive/MyDrive/01_Dev_Projects/Video_Captioning/src/data/DatasetFlickr30k/captions"
else:
    caption_dir = os.path.join(DATA_ROOT, "captions")
    
# --- Encoder Configuration (ViT-Base) ---
vit_cfg = dict(
    image_size=224,      
    patch_size=16,       
    in_channels=3,
    embed_dim=768,       # Base Standard
    depth=12,            
    num_heads=12,        
    mlp_ratio=4.0,       
    dropout=0.1          # Overfit mode: 0.0
)

# --- Decoder Configuration (Transformer) ---
trans_cfg = dict(
    dim=768,             # Matches ViT
    num_heads=12,        
    num_layers=6,        
    ff_dim=3072,         
    dropout=0.1,         
    max_len=40,
    
    # [IMPORTANT] Must match T5 Tokenizer to avoid CUDA Assert errors
    # T5 Base default is 32128. (Training code will update with exact len(tokenizer))
    vocab_size=32128     
)

# --- TRAINING ---
epochs = 15

# --- OPENVINO CONFIGURATION ---
USE_OPENVINO = True
DEVICE = "AUTO" # Options: "CPU", "GPU", "AUTO"

# IR Model Paths
ENCODER_XML = os.path.join(BASE_DIR, "checkpoints", "ir", "captioning", "encoder", "encoder.xml")
DECODER_XML = os.path.join(BASE_DIR, "checkpoints", "ir", "captioning", "decoder", "decoder.xml")
DEPTH_XML = os.path.join(BASE_DIR, "checkpoints", "ir", "depth", "depth_anything_v2_small.xml")
