import torch
import numpy as np
import io
import base64
import os
import gc # Quản lý bộ nhớ
import cv2

from PIL import Image
from transformers import T5Tokenizer
from torchvision import transforms
from deep_translator import GoogleTranslator
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# --- IMPORT MODULE DỰ ÁN ---
# Đảm bảo bạn đã có file config.py và cấu trúc thư mục src/main/...
try:
    from config import vit_cfg, trans_cfg
    from src.main.model.model import ViT_Transformer
    from src.main.distance import DepthEstimator
except ImportError as e:
    print(f"Lỗi Import Module dự án: {e}")
    print("-> Hãy kiểm tra lại cấu trúc thư mục src/main/...")
    exit(1)

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. CẤU HÌNH THIẾT BỊ
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang chạy Server trên thiết bị: {device.upper()}")

# ==========================================
# 2. LOAD MODEL CAPTIONING (ViT-T5)
# ==========================================
print(">> [1/3] Tải Tokenizer...")
try:
    tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.pad_token_id
except Exception as e:
    print(f"Lỗi Tokenizer: {e}")
    exit(1)

print(">> [2/3] Khởi tạo Model ViT-Captioning...")
model_custom = ViT_Transformer(vit_cfg, trans_cfg, vocab_size=len(tokenizer)).to(device)

# --- Load Checkpoint ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "vizwiz_adapted_final.pth")

print(f">> [3/3] Load weights từ: {CHECKPOINT_PATH}")

if not os.path.exists(CHECKPOINT_PATH):
    print(f"CẢNH BÁO: Không tìm thấy file model tại {CHECKPOINT_PATH}")
    # exit(1) # Tạm thời comment để debug nếu chưa có file
else:
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_custom.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model_custom.load_state_dict(checkpoint, strict=False)
        
        model_custom.eval()
        print("Load Model Captioning thành công!")

        # Tối ưu hóa cho CPU (Quantization)
        if device == "cpu":
            print("Đang tối ưu hóa Model (Quantization int8) cho CPU...")
            model_custom = torch.quantization.quantize_dynamic(
                model_custom, 
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            print("Đã nén Model xuống int8!")
    except Exception as e:
        print(f"LỖI LOAD MODEL: {e}")
        exit(1)

# --- Transform ảnh cho Captioning ---
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. LOAD MODEL KHOẢNG CÁCH (Depth Anything)
# ==========================================
print(">> Đang tải Depth Anything V2...")
try:
    # Không cần scale_factor vì đã fix cứng A,B,C trong class
    dist_calc = DepthEstimator(device=device) 
except Exception as e:
    print(f"Lỗi Depth Model: {e}")
    exit(1)

# ==========================================
# 4. HÀM XỬ LÝ PHỤ TRỢ
# ==========================================
def format_distance_output(distance_m, unit_str):
    """Chuyển đổi đơn vị hiển thị"""
    unit_str = unit_str.lower()
    if unit_str == 'm':
        return f"{distance_m:.2f}", ' mét'
    elif unit_str == 'dm':
        return f"{distance_m * 10:.1f}", ' đề-xi-mét'
    else: # Default cm
        return f"{distance_m * 100:.0f}", ' xăng-ti-mét'

# ==========================================
# 5. ROUTE API
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Khai báo biến để cleanup trong finally
    pil_image = None
    cv_image = None
    img_tensor = None
    
    try:
        # A. Nhận dữ liệu
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        unit_pref = data.get('unit', 'cm')
        
        # B. Decode ảnh Base64
        image_data = data['image'].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert sang PIL (cho Captioning) và OpenCV (cho Distance)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        cv_image = np.array(pil_image) 
        # Lưu ý: PIL là RGB, OpenCV mặc định đọc file là BGR. 
        # Nhưng np.array(pil) sẽ ra RGB. DepthAnything và ViT đều nhận RGB tốt.
        # Nếu muốn hiển thị cv2.imshow đúng màu thì mới cần convert BGR.

        # ---------------------------------------------------------
        # C. TÍNH KHOẢNG CÁCH & VỊ TRÍ (UPDATE QUAN TRỌNG)
        # ---------------------------------------------------------
        # Gọi hàm mới trả về 3 giá trị
        distance_meters, position_text, box_coords = dist_calc.estimate_distance(cv_image)
        
        # Format đơn vị (m/cm/dm)
        dist_value, dist_label = format_distance_output(distance_meters, unit_pref)
        dist_display_text = f"{dist_value}{dist_label}"
        
        # ---------------------------------------------------------
        # D. SINH CAPTION (MÔ TẢ ẢNH)
        # ---------------------------------------------------------
        img_tensor = image_transform(pil_image).unsqueeze(0).to(device)
        max_len_cfg = trans_cfg.get('max_len', 30)

        with torch.no_grad():
            caption_en = model_custom.beam_search(
                img_tensor, 
                tokenizer, 
                beam_size=3,             
                max_len=max_len_cfg,     
                device=device,
                no_repeat_ngram_size=2,  
                repetition_penalty=1.0   
            )
        
        # Dịch sang Tiếng Việt
        try:
            caption_vi = GoogleTranslator(source='en', target='vi').translate(caption_en.lower())
        except Exception:
            caption_vi = caption_en # Fallback nếu lỗi mạng

        # ---------------------------------------------------------
        # E. TẠO CÂU NÓI & CẢNH BÁO (LOGIC MỚI)
        # ---------------------------------------------------------
        warning_msg = ""
        
        # Logic ghép câu nói tự nhiên: [Cảnh báo] + [Vị trí] + [Vật thể] + [Khoảng cách]
        if distance_meters < 0.8:
            warning_msg = "CẢNH BÁO! RẤT GẦN"
            # Vd: "Nguy hiểm! Bên trái có Cái ghế. Cách 50 xăng-ti-mét"
            final_speech = f"Nguy hiểm! {position_text} có {caption_vi}. Cách {dist_display_text}"
        elif distance_meters < 1.5:
            final_speech = f"Khá gần. {position_text} là {caption_vi}. Cách {dist_display_text}"
        else:
            final_speech = f"{caption_vi}. {position_text} cách {dist_display_text}"

        # ---------------------------------------------------------
        # F. TRẢ VỀ JSON
        # ---------------------------------------------------------
        response = {
            'caption_vi': caption_vi,
            'distance': dist_display_text,
            'position': position_text,      # Trả về hướng (Trái/Phải/Trước)
            'warning': warning_msg,
            'final_speech': final_speech,
            'unit_used': dist_label,
            'box': box_coords               # Trả về tọa độ [x1, y1, x2, y2] để vẽ
        }
        
        return jsonify(response)

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        return jsonify({'error': str(e)}), 500

    finally:
        # ---------------------------------------------------------
        # G. DỌN DẸP BỘ NHỚ (LUÔN CHẠY)
        # ---------------------------------------------------------
        # Xóa biến
        del pil_image
        del cv_image
        del img_tensor
        
        # Dọn GPU/CPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# ==========================================
# 6. CHẠY SERVER
# ==========================================
# if __name__ == '__main__':
#     from waitress import serve
#     print("\n" + "="*50)
#     print("SERVER ĐANG CHẠY TẠI: http://0.0.0.0:5000")
#     print("Chế độ: Production (Waitress - Multi-thread)")
#     print("="*50 + "\n")
    
#     serve(app, host='0.0.0.0', port=5000, threads=4)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)