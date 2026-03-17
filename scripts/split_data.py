import json
import random
import os
import sys

# --- 1. IMPORT TỪ CONFIG ---
try:
    # Import biến caption_dir từ file config.py
    # Biến này giúp xác định folder đang chứa dữ liệu (Flickr30k hay 8k)
    from config import caption_dir
except ImportError:
    print("Lỗi: Không tìm thấy file config.py. Hãy đặt file này ngang hàng với config.py")
    sys.exit(1)

# --- 2. THIẾT LẬP ĐƯỜNG DẪN TỰ ĐỘNG ---
# Xử lý linh hoạt dù caption_dir là thư mục hay trỏ vào file
if os.path.isdir(caption_dir):
    WORK_DIR = caption_dir
else:
    WORK_DIR = os.path.dirname(caption_dir)

# File đầu vào: Là file JSON tổng vừa tạo từ convert_captions.py
# (Mặc định tên là captions.json)
INPUT_FILE = os.path.join(WORK_DIR, "captions.json")

# Thư mục đầu ra: Lưu ngay tại đó luôn
OUTPUT_DIR = WORK_DIR

def split_dataset(input_path, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Hàm chia file JSON lớn thành 3 file con: Train, Val, Test.
    """
    
    # 1. Kiểm tra file đầu vào
    if not os.path.exists(input_path):
        print(f"LỖI: Không tìm thấy file nguồn tại: {input_path}")
        print("   -> Bạn đã chạy 'convert_captions.py' chưa?")
        return

    print(f"Đang đọc dữ liệu từ: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"LỖI: Không đọc được file JSON. Chi tiết: {e}")
        return
    
    # Lấy danh sách ảnh (Xử lý linh hoạt)
    # File convert_captions.py của chúng ta tạo ra List of Dicts, nên thường sẽ rơi vào else hoặc elif
    if isinstance(data, dict) and 'images' in data:
        images = data['images']
    elif isinstance(data, list):
        images = data
    else:
        print("LỖI: Cấu trúc JSON không hợp lệ (không tìm thấy list ảnh).")
        return

    # 2. Xáo trộn dữ liệu (Shuffle)
    print("Đang xáo trộn dữ liệu...")
    random.seed(42) 
    random.shuffle(images)
    
    # 3. Tính toán số lượng
    total = len(images)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    # n_test là phần còn lại
    
    train_data = images[:n_train]
    val_data = images[n_train : n_train + n_val]
    test_data = images[n_train + n_val:]
    
    print(f"Tổng số ảnh tìm thấy: {total}")
    print(f"   - Train (80%): {len(train_data)} ảnh")
    print(f"   - Val   (10%): {len(val_data)} ảnh")
    print(f"   - Test  (10%): {len(test_data)} ảnh")
    
    # 4. Tạo thư mục đầu ra nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Hàm hỗ trợ lưu file
    def save_json(data_list, filename):
        file_path = os.path.join(output_dir, filename)
        
        # Nếu data gốc có cấu trúc lồng, giữ nguyên cấu trúc đó
        if isinstance(data, dict) and 'images' in data:
            final_content = {'images': data_list}
            for k, v in data.items():
                if k != 'images':
                    final_content[k] = v
        else:
            final_content = data_list # List phẳng
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(final_content, f, ensure_ascii=False, indent=2)
        print(f"   -> Đã lưu: {filename}")

    # Thực hiện lưu
    print("Đang lưu 3 file con...")
    save_json(train_data, 'train_captions.json')
    save_json(val_data, 'val_captions.json')
    save_json(test_data, 'test_captions.json')
    
    print(f"\nHOÀN TẤT! Các file đã sẵn sàng tại: {output_dir}")
    print("   Bây giờ bạn có thể chạy 'python train.py'!")

# --- MAIN ---
if __name__ == "__main__":
    print(f"Config đang trỏ tới: {WORK_DIR}")
    
    # Nếu file captions.json không tồn tại, thử tìm tên cũ captions.json
    if not os.path.exists(INPUT_FILE):
        fallback_file = os.path.join(WORK_DIR, "captions.json")
        if os.path.exists(fallback_file):
            print(f"Không thấy 'captions.json', đang dùng 'captions.json'")
            INPUT_FILE = fallback_file
            
    split_dataset(INPUT_FILE, OUTPUT_DIR)