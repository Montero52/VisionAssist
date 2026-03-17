import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CẤU HÌNH ĐƯỜNG DẪN THÔNG MINH ---
# 1. Ưu tiên dữ liệu tạm trên Colab (Tốc độ cao)
if os.path.exists("/content/temp_data/flickr30k_images"):
    DATA_ROOT = "/content/temp_data/flickr30k_images"
    print(f"Dataset Root: {DATA_ROOT}")
    
    # [QUAN TRỌNG] Kiểm tra xem ảnh nằm ở đâu
    # Trường hợp 1: flickr30k_images/flickr30k_images/*.jpg (Thường gặp khi unzip)
    if os.path.exists(os.path.join(DATA_ROOT, "flickr30k_images")):
        image_dir = os.path.join(DATA_ROOT, "flickr30k_images")
    # Trường hợp 2: flickr30k_images/images/*.jpg (Code cũ của bạn)
    elif os.path.exists(os.path.join(DATA_ROOT, "images")):
        image_dir = os.path.join(DATA_ROOT, "images")
    # Trường hợp 3: Ảnh nằm ngay trong DATA_ROOT
    else:
        image_dir = DATA_ROOT
        
    print(f"Đã tự động dò tìm thư mục ảnh tại: {image_dir}")

# 2. Chạy trên Drive
elif os.path.exists("/content/drive/MyDrive/01_Dev_Projects/Video_Captioning/src/data/DatasetFlickr30k"):
    DATA_ROOT = "/content/drive/MyDrive/01_Dev_Projects/Video_Captioning/src/data/DatasetFlickr30k"
    image_dir = os.path.join(DATA_ROOT, "images") # Trên Drive bạn tự quản lý nên chắc là đúng
    print(f"Đang chạy Drive: {image_dir}")

# 3. Chạy Local
else:
    DATA_ROOT = os.path.join(BASE_DIR, "src", "data", "flickr8k")
    image_dir = os.path.join(DATA_ROOT, "images")
    print(f"Đang chạy Local: {image_dir}")

# Cấu hình đường dẫn Caption (Giữ nguyên)
if "temp_data" in DATA_ROOT:
    caption_dir = "/content/drive/MyDrive/01_Dev_Projects/Video_Captioning/src/data/DatasetFlickr30k/captions"
else:
    caption_dir = os.path.join(DATA_ROOT, "captions")
    
# --- Cấu hình Encoder (ViT-Base) ---
vit_cfg = dict(
    image_size=224,      
    patch_size=16,       
    in_channels=3,
    embed_dim=768,       # Chuẩn Base
    depth=12,            
    num_heads=12,        
    mlp_ratio=4.0,       
    dropout=0.1          # Overfit mode: 0.0
)

# --- Cấu hình Decoder (Transformer) ---
trans_cfg = dict(
    dim=768,             # Khớp với ViT
    num_heads=12,        
    num_layers=6,        
    ff_dim=3072,         
    dropout=0.1,         
    max_len=40,
    
    # [QUAN TRỌNG] Phải khớp với T5 Tokenizer để tránh lỗi CUDA Assert
    # T5 Base mặc định là 32128. (Code train sẽ cập nhật lại số chính xác từ len(tokenizer))
    vocab_size=32128     
)

# --- HUẤN LUYỆN ---
epochs = 15