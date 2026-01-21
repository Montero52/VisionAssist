import json
import os
import csv
import sys

# --- 1. IMPORT TỪ CONFIG ---
try:
    # Import biến caption_dir từ file config.py
    from config import caption_dir
except ImportError:
    print("❌ Lỗi: Không tìm thấy file config.py. Hãy đặt file này ngang hàng với config.py")
    sys.exit(1)

# --- 2. THIẾT LẬP ĐƯỜNG DẪN TỰ ĐỘNG ---

# File JSON đích (Chính là đường dẫn config.py đang dùng để train)
OUTPUT_JSON_FILE = caption_dir

# File TXT nguồn (Suy luận: Nằm cùng thư mục với file JSON, tên là captions.txt)
# os.path.dirname(caption_dir) -> Lấy thư mục chứa file (ví dụ: .../flickr8k/captions)
folder_path = os.path.dirname(OUTPUT_JSON_FILE)
INPUT_TXT_FILE = os.path.join(folder_path, "captions.txt")

def convert_txt_to_json():
    print(f"📂 Thư mục làm việc: {folder_path}")
    
    # Kiểm tra file nguồn
    if not os.path.exists(INPUT_TXT_FILE):
        print(f"❌ LỖI: Không tìm thấy file nguồn tại:\n   {INPUT_TXT_FILE}")
        print("👉 Hãy chắc chắn bạn đã đổi tên file gốc thành 'captions.txt' và để trong thư mục captions.")
        return

    print(f"🔄 Đang đọc file nguồn: {os.path.basename(INPUT_TXT_FILE)}...")
    
    temp_dict = {}
    count_skipped = 0

    try:
        with open(INPUT_TXT_FILE, 'r', encoding='utf-8') as f:
            # Dữ liệu Flickr8k gốc thường ngăn cách bằng dấu phẩy
            # Nếu file txt của bạn ngăn cách bằng tab (\t) hay gì khác, hãy sửa delimiter
            reader = csv.reader(f, delimiter=',') 
            
            # Thử bỏ qua header (nếu dòng đầu tiên là image,caption)
            first_line = next(reader, None)
            if first_line and "image" not in first_line[0].lower():
                # Nếu dòng đầu không phải header, quay lại từ đầu
                f.seek(0)
                reader = csv.reader(f, delimiter=',')

            for row in reader:
                if len(row) < 2:
                    count_skipped += 1
                    continue
                
                img_name = row[0].strip()
                # Gom các phần còn lại thành caption (đề phòng caption có chứa dấu phẩy)
                caption = ",".join(row[1:]).strip()
                
                # Xử lý tên ảnh kiểu cũ (image.jpg#0)
                if "#" in img_name:
                    img_name = img_name.split("#")[0]

                if img_name not in temp_dict:
                    temp_dict[img_name] = []
                
                # Chỉ thêm caption nếu chưa có (tránh trùng lặp)
                if caption not in temp_dict[img_name]:
                    temp_dict[img_name].append(caption)

    except Exception as e:
        print(f"⚠️ Lỗi đọc file: {e}")
        return

    # Chuyển đổi sang format list object
    final_data = []
    for img, caps in temp_dict.items():
        entry = {
            "file_name": img,
            "captions": caps
        }
        final_data.append(entry)

    print(f"✅ Đã xử lý {len(final_data)} ảnh.")
    if count_skipped > 0:
        print(f"⚠️ Đã bỏ qua {count_skipped} dòng lỗi/trống.")

    # Tạo thư mục đích nếu chưa có (phòng hờ)
    os.makedirs(os.path.dirname(OUTPUT_JSON_FILE), exist_ok=True)

    # Lưu file JSON
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    print(f"🎉 XONG! File JSON chuẩn đã được lưu tại:\n   {OUTPUT_JSON_FILE}")
    print("👉 Bây giờ bạn có thể chạy 'python train.py' được rồi!")

if __name__ == "__main__":
    convert_txt_to_json()