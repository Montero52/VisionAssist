import os
import gdown

# 1. Cấu hình ID file Google Drive (Thay ID của bạn vào đây)
FILE_ID = '1Dv-X56iR1E3DLqKSZXn5didC-0gpKNZZ'

# 2. Cấu hình đường dẫn lưu
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_FILE = os.path.join(CHECKPOINT_DIR, "model_epoch_50.pth")

# 3. Hàm tải
def download_model():
    # Tạo thư mục checkpoints nếu chưa có
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        print(f"Đã tạo thư mục: {CHECKPOINT_DIR}")

    # Kiểm tra nếu file đã có rồi thì thôi
    if os.path.exists(OUTPUT_FILE):
        print(f"File model đã tồn tại tại: {OUTPUT_FILE}")
        return

    print(f"⬇Đang tải model từ Google Drive... (Vui lòng chờ)")
    
    # URL tải của gdown
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    
    try:
        gdown.download(url, OUTPUT_FILE, quiet=False)
        print(f"\nTải thành công! File đã được lưu vào: {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nLỗi tải file: {e}")
        print("Vui lòng tải thủ công theo link trong README.md")

if __name__ == '__main__':
    download_model()