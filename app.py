import torch
import numpy as np
import io
import base64
import gc
import logging
import time
import threading

from collections import OrderedDict
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
try:
    import config
    from src.main.distance import DepthEstimator
except ImportError as e:
    logger.error(f"Project Module Import Error: {e}")
    logger.error("-> Please check src/main/ directory structure...")
    exit(1)

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. TRANSLATION CACHE
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
# 2. LOAD MODELS (OPENVINO ONLY)
# ==========================================
logger.info(">> [1/2] Loading Tokenizer...")
try:
    tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.pad_token_id
except Exception as e:
    logger.error(f"Tokenizer Error: {e}")
    exit(1)

logger.info(">> [2/2] Initializing Depth & Captioning Models via OpenVINO Engine...")
try:
    # We use DepthEstimator as the master model holder because it integrates 
    # both distance calculation logic and the OpenVINOEngine.
    # OpenVINOEngine internally loads Encoder, Decoder, and Depth models.
    dist_calc = DepthEstimator(device="AUTO", use_openvino=True)
    ov_engine = dist_calc.ov_engine # Shared engine for Captioning
    logger.info("OpenVINO Models loaded successfully!")
except Exception as e:
    logger.error(f"FAILED TO LOAD OPENVINO MODELS: {e}")
    exit(1)

# --- Image Transforms for Captioning ---
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.685, 0.656, 0.606], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. WARMUP & MEMORY CLEANUP
# ==========================================
def warmup_and_cleanup():
    """Warms up models with dummy data and clears PyTorch RAM."""
    try:
        logger.info(">> Warming up OpenVINO models (first inference)...")
        dummy_np = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Warmup Depth & Distance logic
        _ = dist_calc.estimate_distance(dummy_np)
        
        # Warmup Captioning (Encoder + Beam Search)
        dummy_pil = Image.fromarray(dummy_np, mode="RGB")
        dummy_tensor = image_transform(dummy_pil).unsqueeze(0).numpy()
        _ = ov_engine.beam_search(
            dummy_tensor,
            tokenizer,
            beam_size=1,
            max_len=10
        )
        
        logger.info(">> Warmup completed. OpenVINO is ready.")
        
        # Explicitly clear any initial PyTorch overhead
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        logger.warning(f">> Warmup skipped: {e}")

warmup_and_cleanup()

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

model_lock = threading.Lock()

@app.route('/predict', methods=['POST'])
def predict():
    with model_lock:
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
            
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            cv_image = np.array(pil_image) 

            # ---------------------------------------------------------
            # C. DISTANCE & POSITION ESTIMATION (OpenVINO)
            # ---------------------------------------------------------
            t_dist0 = time.perf_counter()
            distance_meters, position_text, box_coords = dist_calc.estimate_distance(cv_image)
            t_dist1 = time.perf_counter()
            
            # Log distance inference time
            logger.info(f"OpenVINO Distance Inference Time: {(t_dist1 - t_dist0)*1000:.2f}ms")
            
            # Format units
            dist_value, dist_label = format_distance_output(distance_meters, unit_pref)
            dist_display_text = f"{dist_value}{dist_label}"
            
            caption_en = ""
            caption_text = ""

            # ---------------------------------------------------------
            # D. IMAGE CAPTIONING (OpenVINO) - ONLY IF mode=full
            # ---------------------------------------------------------
            if mode == "full":
                t_cap0 = time.perf_counter()
                img_tensor = image_transform(pil_image).unsqueeze(0).numpy()
                
                # Use OpenVINO Engine for Beam Search
                caption_en = ov_engine.beam_search(
                    img_tensor,
                    tokenizer,
                    beam_size=1, # Greedy search for maximum performance on CPU
                    max_len=25
                )

                if lang == "en":
                    caption_text = (caption_en or "").strip()
                else:
                    caption_text = translate_en_to_vi_cached(caption_en)
                    
                t_cap1 = time.perf_counter()
                logger.info(f"OpenVINO Caption Inference Time: {(t_cap1 - t_cap0)*1000:.2f}ms")
            else:
                t_cap0 = t_cap1 = None

            # ---------------------------------------------------------
            # E. SPEECH GENERATION & WARNINGS
            # ---------------------------------------------------------
            warning_msg = ""
            if distance_meters < 0.6:
                warning_msg = "CẢNH BÁO! RẤT GẦN" if lang == "vi" else "WARNING! VERY CLOSE"
                danger_word = "Nguy hiểm!" if lang == "vi" else "Danger!"
            elif distance_meters < 1.5:
                danger_word = "Khá gần." if lang == "vi" else "Nearby."
            else:
                danger_word = ""

            if mode == "full":
                if lang == "vi":
                    final_speech = f"{danger_word} {position_text} có {caption_text}. Cách {dist_display_text}"
                else:
                    final_speech = f"{danger_word} {position_text} has {caption_text}. {dist_display_text}."
            else:
                if lang == "vi":
                    final_speech = f"{danger_word} {position_text}. Cách {dist_display_text}."
                else:
                    final_speech = f"{danger_word} {position_text}. {dist_display_text}."

            # ---------------------------------------------------------
            # F. RETURN JSON RESPONSE
            # ---------------------------------------------------------
            response = {
                'caption_vi': caption_text,
                'distance': dist_display_text,
                'position': position_text,      
                'warning': warning_msg,
                'final_speech': final_speech,
                'unit_used': dist_label,
                'box': box_coords,
                'timing_ms': {
                    "total": round((time.perf_counter() - t0) * 1000, 1),
                    "distance": round((t_dist1 - t_dist0) * 1000, 1),
                    "caption": round((t_cap1 - t_cap0) * 1000, 1) if mode == "full" else 0.0
                }
            }
            
            return jsonify(response)

        except Exception as e:
            logger.error(f"SERVER ERROR: {e}")
            return jsonify({'error': str(e)}), 500

        finally:
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
    logger.info("Mode: Production (OpenVINO + Waitress)")
    logger.info("="*50 + "\n")
    
    serve(app, host='0.0.0.0', port=5000, threads=4)
