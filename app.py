import torch
import numpy as np
import io
import base64
import os
import gc
import cv2
import logging
from collections import OrderedDict
import time

from PIL import Image
from transformers import T5Tokenizer
from torchvision import transforms
from deep_translator import GoogleTranslator
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("visionassist.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("VisionAssist-Server")

# --- PROJECT MODULE IMPORTS ---
# Ensure config.py exists and src/main/... directory structure is correct
try:
    from config import vit_cfg, trans_cfg
    from src.main.model.model import ViT_Transformer
    from src.main.distance import DepthEstimator
except ImportError as e:
    logger.error(f"Project Module Import Error: {e}")
    logger.error("-> Please check src/main/ directory structure...")
    exit(1)

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. DEVICE CONFIGURATION
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Server running on device: {device.upper()}")

# ==========================================
# 1.5 TRANSLATION CACHE (REDUCE LATENCY + INCREASE STABILITY)
# ==========================================
TRANSLATION_CACHE_MAX = 256
translation_cache = OrderedDict()
request_counter = 0

def translate_en_to_vi_cached(text_en: str) -> str:
    """Translates English text to Vietnamese with caching."""
    key = (text_en or "").strip().lower()
    if not key:
        return text_en

    cached = translation_cache.get(key)
    if cached is not None:
        translation_cache.move_to_end(key)
        return cached

    try:
        vi = GoogleTranslator(source="en", target="vi").translate(key)
        translation_cache[key] = vi
        translation_cache.move_to_end(key)
        if len(translation_cache) > TRANSLATION_CACHE_MAX:
            translation_cache.popitem(last=False)
        return vi
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text_en

# ==========================================
# 2. LOAD CAPTIONING MODEL (ViT-T5)
# ==========================================
logger.info(">> [1/3] Loading Tokenizer...")
try:
    tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.pad_token_id
except Exception as e:
    logger.error(f"Tokenizer Error: {e}")
    exit(1)

logger.info(">> [2/3] Initializing ViT-Captioning Model...")
model_custom = ViT_Transformer(vit_cfg, trans_cfg, vocab_size=len(tokenizer)).to(device)

# --- Load Checkpoint ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "vizwiz_adapted_final.pth")

logger.info(f">> [3/3] Loading weights from: {CHECKPOINT_PATH}")

if not os.path.exists(CHECKPOINT_PATH):
    logger.warning(f"WARNING: Model file not found at {CHECKPOINT_PATH}")
    # exit(1) # Temporarily commented for debug
else:
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_custom.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model_custom.load_state_dict(checkpoint, strict=False)
        
        model_custom.eval()
        logger.info("Captioning Model loaded successfully!")

        # CPU Optimization (Quantization)
        if device == "cpu":
            logger.info("Optimizing Model (INT8 Quantization) for CPU...")
            model_custom = torch.quantization.quantize_dynamic(
                model_custom, 
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            logger.info("Model compressed to INT8!")
    except Exception as e:
        logger.error(f"LOAD MODEL ERROR: {e}")
        exit(1)

# --- Image Transforms for Captioning ---
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.685, 0.656, 0.606], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. LOAD DISTANCE MODEL (Depth Anything V2)
# ==========================================
logger.info(">> Loading Depth Anything V2...")
try:
    # A, B, C constants are fixed in the class
    dist_calc = DepthEstimator(device=device) 
except Exception as e:
    logger.error(f"Depth Model Error: {e}")
    exit(1)

# ==========================================
# 3.5 WARMUP (REDUCE FIRST REQUEST JITTER ON CPU)
# ==========================================
def warmup_models():
    """Warms up models with a dummy image."""
    try:
        logger.info(">> Warming up models...")
        dummy_np = np.zeros((240, 320, 3), dtype=np.uint8)
        dummy_pil = Image.fromarray(dummy_np, mode="RGB")

        # Warmup Depth
        _ = dist_calc.estimate_distance(dummy_np)

        # Warmup Caption (no translation)
        dummy_tensor = image_transform(dummy_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model_custom.beam_search(
                dummy_tensor,
                tokenizer,
                beam_size=1,
                max_len=10,
                device=device,
                no_repeat_ngram_size=2,
                repetition_penalty=1.0
            )

        logger.info(">> Warmup completed.")
    except Exception as e:
        logger.warning(f">> Warmup skipped: {e}")

warmup_models()

# ==========================================
# 4. UTILITY FUNCTIONS
# ==========================================
def format_distance_output(distance_m, unit_str):
    """Converts distance to target unit string."""
    unit_str = unit_str.lower()
    if unit_str == 'm':
        return f"{distance_m:.2f}", ' mét'
    elif unit_str == 'dm':
        return f"{distance_m * 10:.1f}", ' đề-xi-mét'
    else: # Default cm
        return f"{distance_m * 100:.0f}", ' xăng-ti-mét'

# ==========================================
# 5. API ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Variables for cleanup in finally block
    pil_image = None
    cv_image = None
    img_tensor = None
    
    try:
        global request_counter
        request_counter += 1
        t0 = time.perf_counter()

        # A. Receive Data
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        unit_pref = data.get('unit', 'cm')
        lang = (data.get('lang', 'vi') or 'vi').lower()  # "vi" | "en"
        mode = (data.get('mode', 'full') or 'full').lower()  # "full" | "distance_only"
        
        # B. Decode Base64 Image
        image_data = data['image'].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL (for Captioning) and OpenCV (for Distance)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        cv_image = np.array(pil_image) 
        # Note: PIL is RGB, OpenCV defaults to BGR for file reading.
        # But np.array(pil) yields RGB. DepthAnything and ViT both accept RGB.

        # ---------------------------------------------------------
        # C. DISTANCE & POSITION ESTIMATION
        # ---------------------------------------------------------
        t_dist0 = time.perf_counter()
        with torch.no_grad():
            distance_meters, position_text, box_coords = dist_calc.estimate_distance(cv_image)
        t_dist1 = time.perf_counter()
        
        # Log distance inference time
        logger.info(f"Distance Inference Time: {(t_dist1 - t_dist0)*1000:.2f}ms")
        
        # Format units (m/cm/dm)
        dist_value, dist_label = format_distance_output(distance_meters, unit_pref)
        dist_display_text = f"{dist_value}{dist_label}"
        
        caption_en = ""
        caption_text = ""  # language based on lang: vi or en

        # ---------------------------------------------------------
        # D. IMAGE CAPTIONING - ONLY IF mode=full
        # ---------------------------------------------------------
        if mode == "full":
            t_cap0 = time.perf_counter()
            img_tensor = image_transform(pil_image).unsqueeze(0).to(device)
            # CPU-friendly max length
            max_len_cfg = min(25, int(trans_cfg.get('max_len', 40)))

            with torch.no_grad():
                caption_en = model_custom.beam_search(
                    img_tensor,
                    tokenizer,
                    beam_size=1,
                    max_len=max_len_cfg,
                    device=device,
                    no_repeat_ngram_size=2,
                    repetition_penalty=1.0
                )

            if lang == "en":
                caption_text = (caption_en or "").strip()
            else:
                try:
                    caption_text = translate_en_to_vi_cached(caption_en)
                except Exception:
                    caption_text = caption_en  # fallback on network error
            t_cap1 = time.perf_counter()
            
            # Log caption inference time
            logger.info(f"Caption Inference Time: {(t_cap1 - t_cap0)*1000:.2f}ms")
        else:
            caption_text = ""
            t_cap0 = None
            t_cap1 = None

        # ---------------------------------------------------------
        # E. SPEECH GENERATION & WARNINGS
        # ---------------------------------------------------------
        warning_msg = ""

        if mode != "full":
            # Distance-only: direction + distance + warning
            if lang == "en":
                if distance_meters < 0.6:
                    warning_msg = "WARNING! VERY CLOSE"
                    final_speech = f"Danger! {position_text}. {dist_display_text}."
                elif distance_meters < 1.5:
                    final_speech = f"Nearby. {position_text}. {dist_display_text}."
                else:
                    final_speech = f"{position_text}. {dist_display_text}."
            else:
                if distance_meters < 0.6:
                    warning_msg = "CẢNH BÁO! RẤT GẦN"
                    final_speech = f"Nguy hiểm! {position_text}. Cách {dist_display_text}."
                elif distance_meters < 1.5:
                    final_speech = f"Khá gần. {position_text}. Cách {dist_display_text}."
                else:
                    final_speech = f"{position_text}. Cách {dist_display_text}."
        else:
            # Full: includes caption
            if lang == "en":
                if distance_meters < 0.6:
                    warning_msg = "WARNING! VERY CLOSE"
                    final_speech = f"Danger! {position_text} has {caption_text}. {dist_display_text}."
                elif distance_meters < 1.5:
                    final_speech = f"Nearby. {position_text} is {caption_text}. {dist_display_text}."
                else:
                    final_speech = f"{caption_text}. {position_text}. {dist_display_text}."
            else:
                if distance_meters < 0.6:
                    warning_msg = "CẢNH BÁO! RẤT GẦN"
                    final_speech = f"Nguy hiểm! {position_text} có {caption_text}. Cách {dist_display_text}"
                elif distance_meters < 1.5:
                    final_speech = f"Khá gần. {position_text} là {caption_text}. Cách {dist_display_text}"
                else:
                    final_speech = f"{caption_text}. {position_text} cách {dist_display_text}"

        # ---------------------------------------------------------
        # F. RETURN JSON RESPONSE
        # ---------------------------------------------------------
        response = {
            'caption_vi': caption_text,
            'distance': dist_display_text,
            'position': position_text,      # Direction (Left/Right/Ahead)
            'warning': warning_msg,
            'final_speech': final_speech,
            'unit_used': dist_label,
            'box': box_coords               # Bounding box coords [x1, y1, x2, y2]
        }

        # Timing info
        t1 = time.perf_counter()
        response["timing_ms"] = {
            "total": round((t1 - t0) * 1000, 1),
            "distance": round((t_dist1 - t_dist0) * 1000, 1),
            "caption": round((t_cap1 - t_cap0) * 1000, 1) if (t_cap0 is not None and t_cap1 is not None) else 0.0
        }
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"SERVER ERROR: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        # ---------------------------------------------------------
        # G. MEMORY CLEANUP
        # ---------------------------------------------------------
        # Avoid deleting variables not assigned due to early exceptions
        pil_image = None
        cv_image = None
        img_tensor = None

        # Periodically trigger garbage collection
        if request_counter % 50 == 0:
            gc.collect()

# ==========================================
# 6. RUN SERVER
# ==========================================
if __name__ == '__main__':
    from waitress import serve
    logger.info("\n" + "="*50)
    logger.info("SERVER RUNNING AT: http://127.0.0.1:5000")
    logger.info("Mode: Production (Waitress - Multi-thread)")
    logger.info("="*50 + "\n")
    
    serve(app, host='0.0.0.0', port=5000, threads=4)
