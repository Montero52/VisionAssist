import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CẤU HÌNH ĐƯỜNG DẪN THÔNG MINH ---
# 1. Ưu tiên dữ liệu tạm trên Colab (Tốc độ cao)
if os.path.exists("/content/temp_data/images"):
    DATA_ROOT = "/content/temp_data/images"
    print(f"Đang chạy trên Colab với dữ liệu tốc độ cao tại: {DATA_ROOT}")

# 2. Nếu không có, thử tìm trên Drive (Tốc độ chậm hơn)
elif os.path.exists("/content/drive/MyDrive/Video Captioning/src/data/flickr8k"):
    DATA_ROOT = "/content/drive/MyDrive/Video Captioning/src/data/flickr8k"
    print(f"Đang chạy trực tiếp trên Drive (Sẽ hơi chậm): {DATA_ROOT}")

# 3. Cuối cùng là chạy Local (VS Code)
else:
    DATA_ROOT = os.path.join(BASE_DIR, "src", "data", "flickr8k")
    print(f"Đang chạy Local tại: {DATA_ROOT}")

# Các đường dẫn con (Giữ nguyên)
image_dir = os.path.join(DATA_ROOT, "images")
# Lưu ý: File json vẫn đọc từ Drive hoặc Local vì nó nhẹ, không cần unzip
if "temp_data" in DATA_ROOT:
    # Nếu đang dùng temp_data, ta trỏ caption về lại folder code gốc để đỡ phải copy file json
    # Giả sử cấu trúc unzip xong chỉ có folder Images
    caption_dir = "/content/drive/MyDrive/Video Captioning/src/data/flickr8k/captions"
else:
    caption_dir = os.path.join(DATA_ROOT, "captions")

vit_cfg = dict(
    image_size=224,      # Kích thước ảnh đầu vào (Cố định)
    patch_size=16,       # Chia ảnh thành các ô 16x16 (Chi tiết tốt hơn 32)
    in_channels=3,       # Ảnh màu RGB
    embed_dim=512,       # Kích thước vector đặc trưng (Medium size)
    depth=6,             # Số lớp Encoder (6 lớp là đủ cho 5GB dữ liệu)
    num_heads=8,         # Số lượng đầu Attention (512 / 8 = 64 dim/head -> OK)
    mlp_ratio=4.0,       # Hệ số mở rộng trong lớp FeedForward
    dropout=0.1          # Giảm Overfitting
)

# --- Cấu hình Decoder (Transformer) ---
trans_cfg = dict(
    dim=512,             # Kích thước vector đầu vào (Khớp với ViT để ko cần Project)
    num_heads=8,         # Khớp với ViT
    num_layers=6,        # Số lớp Decoder (Cân bằng với Encoder)
    ff_dim=2048,         # Thường gấp 4 lần dim (512 * 4)
    dropout=0.1,         
    max_len=40           # Độ dài tối đa của câu caption (Flickr30k câu khá dài)
)
epochs = 20