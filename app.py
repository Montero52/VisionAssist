from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
from transformers import T5Tokenizer
from torchvision import transforms
# --- THAY ĐỔI: Dùng deep_translator thay vì googletrans ---
from deep_translator import GoogleTranslator
import io
import base64
import os   

# --- IMPORT CÁC MODULE CỦA DỰ ÁN ---
try:
    from config import vit_cfg, trans_cfg
    from src.main.model.model import ViT_Transformer
    # Lưu ý: Import từ file distance gốc mà bạn đã tối giản
    from src.main.distance import DistanceCalculator 
except ImportError as e:
    print("LỖI IMPORT: Vui lòng kiểm tra cấu trúc thư mục.")
    print(f"Chi tiết: {e}")
    exit(1)

app = Flask(__name__)
CORS(app)

# Tự động chọn thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- Đang chạy trên thiết bị: {device.upper()} ---")

# ==========================================
# 1. LOAD MODEL CAPTIONING (CUSTOM VIT-T5)
# ==========================================
print(">> Đang tải Tokenizer (T5-base)...")
try:
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.pad_token_id
except Exception as e:
    print(f"Lỗi tải Tokenizer: {e}")
    exit(1)

print(">> Đang khởi tạo Model ViT_Transformer...")
model_custom = ViT_Transformer(vit_cfg, trans_cfg, vocab_size=len(tokenizer)).to(device)

# --- Cấu hình đường dẫn model chuẩn ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
CHECKPOINT_NAME = "model_epoch_50.pth"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

print(f">> Đang nạp trọng số từ: {CHECKPOINT_PATH}")

# Kiểm tra file model
if not os.path.exists(CHECKPOINT_PATH):
    print("\n" + "="*50)
    print(f"LỖI NGHIÊM TRỌNG: Không tìm thấy file model!")
    print(f"   Đường dẫn mong muốn: {CHECKPOINT_PATH}")
    print("="*50 + "\n")
    exit(1)

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_custom.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model_custom.load_state_dict(checkpoint, strict=False)
    
    model_custom.eval()
    print("Load Custom Model thành công!")
except Exception as e:
    print(f"LỖI LOAD MODEL: {e}")
    exit(1)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================================
# 2. LOAD MODEL KHOẢNG CÁCH (DPT_HYBRID)
# ==========================================
print(">> Đang tải MiDaS (DPT_Hybrid)...")
try:
    midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True) 
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).dpt_transform
except Exception as e:
    print(f"CẢNH BÁO: DPT_Hybrid lỗi. Thử MiDaS_small...")
    try:
        midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    except Exception as e2:
         print(f"LỖI MẠNG: {e2}")
         exit(1)

midas_model.to(device)
midas_model.eval()

# ==========================================
# CẤU HÌNH TÍNH KHOẢNG CÁCH (ĐÃ TỐI GIẢN)
# ==========================================
# Chỉ cần 1 thông số duy nhất: Scale Factor
MY_SCALE_FACTOR = 1880.0  # đợi 2: 415 * 1/0.22    đợt 1: #1200.0 * 1/2.89

dist_calc = DistanceCalculator(
    midas_model, 
    midas_transforms, 
    device, 
    scale_factor=MY_SCALE_FACTOR
)

# ==========================================
# HÀM HỖ TRỢ
# ==========================================
def format_distance_output(distance_m, unit_str):
    unit_str = unit_str.lower()
    if unit_str == 'm':
        value = distance_m
        unit_label = ' mét'
        precision = 2
    elif unit_str == 'dm':
        value = distance_m * 10
        unit_label = ' đề-xi-mét'
        precision = 1
    else: # cm
        value = distance_m * 100
        unit_label = ' xăng-ti-mét'
        precision = 0 
    
    return f"{value:.{precision}f}", unit_label

# ==========================================
# 3. API ROUTE
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Nhận dữ liệu
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'Không tìm thấy ảnh gửi lên'}), 400
            
        unit_pref = data.get('unit', 'cm')
            
        image_data = data['image'].split(",")[1]
        image_bytes = base64.b64decode(image_data)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        cv_image = np.array(pil_image)

        # 2. Đo khoảng cách (Chỉ lấy Distance Z)
        # Hàm estimate_distance mới (trong file distance.py tối giản) chỉ trả về 1 số float
        distance_meters = dist_calc.estimate_distance(cv_image)
        
        # 3. Format đơn vị
        dist_value, dist_unit_label = format_distance_output(distance_meters, unit_pref)
        dist_text = f"{dist_value}{dist_unit_label}"
        
        # 4. Logic cảnh báo (Dựa hoàn toàn vào Z)
        warning_msg = ""
        prefix_speech = "Phía trước là "
        
        # Nếu < 0.8m: Cảnh báo nguy hiểm
        if distance_meters < 0.8:
            warning_msg = "CẨN THẬN! RẤT GẦN"
            prefix_speech = "Nguy hiểm! Ngay trước mặt là "
        # Nếu < 1.5m: Cảnh báo nhẹ
        elif distance_meters < 1.5:
            prefix_speech = "Khá gần, phía trước có "

        # 5. Sinh Caption
        img_tensor = image_transform(pil_image).unsqueeze(0).to(device)
        max_len_caption = trans_cfg.get('max_len', 40) 

        with torch.no_grad():
            caption_en = model_custom.generate(
                img_tensor, 
                tokenizer, 
                max_len=max_len_caption,
                device=device,
                temperature=1.0,
                top_k=5,
                top_p=0.9
            )
        
        # 6. Dịch sang tiếng Việt
        try:
            caption_vi = GoogleTranslator(source='en', target='vi').translate(caption_en)
        except Exception:
            caption_vi = caption_en 

        # 7. Tạo câu nói cuối cùng
        final_speech = f"{prefix_speech} {caption_vi} cách {dist_value} {dist_unit_label}"

        return jsonify({
            'caption_vi': caption_vi,
            'distance': dist_text,
            'warning': warning_msg,
            'final_speech': final_speech,
            'unit_used': dist_unit_label
        })

    except Exception as e:
        print(f"LỖI SERVER: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)