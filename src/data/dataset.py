import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import random

class JsonCaptionsDataset(Dataset):
    def __init__(self, root, annFile, image_transform=None, caption_tokenizer=None,
                 max_len=64, img_key="file_name", cap_key="captions"):
        """
        Args:
            root (str): Đường dẫn folder chứa ảnh.
            annFile (str): Đường dẫn file JSON chứa caption.
            image_transform: Hàm biến đổi ảnh (Resize, Normalize...).
            caption_tokenizer: Tokenizer (T5Tokenizer) để mã hóa văn bản.
            max_len (int): Độ dài tối đa của caption.
        """
        self.root = root
        self.image_transform = image_transform
        self.caption_tokenizer = caption_tokenizer
        self.max_len = max_len
        self.img_key = img_key
        self.cap_key = cap_key

        # 1. Đọc file JSON an toàn
        try:
            with open(annFile, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Lỗi không đọc được file JSON: {annFile}\nChi tiết: {e}")

        # 2. Xử lý format JSON (List phẳng hoặc Dict)
        records = data["images"] if isinstance(data, dict) and "images" in data else data

        self.items = []
        for d in records:
            fn = d.get(self.img_key)
            caps = d.get(self.cap_key, [])
            
            # Bỏ qua nếu dữ liệu thiếu tên file hoặc không có caption
            if not fn or not caps:
                continue

            # Làm sạch caption (chỉ lấy string)
            caps = [str(c).strip() for c in caps if isinstance(c, (str, list))]
            
            if len(caps) > 0:
                self.items.append((os.path.join(root, fn), caps))

        if not self.items:
            raise RuntimeError("Không tìm thấy dữ liệu hợp lệ nào! Hãy kiểm tra lại cấu trúc file JSON.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, caps = self.items[idx]

        # --- A. XỬ LÝ ẢNH ---
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new('RGB', (224, 224), color='black')

        if self.image_transform:
            img = self.image_transform(img)

        # --- B. XỬ LÝ CAPTION ---
        caption_text = random.choice(caps)

        if self.caption_tokenizer:
            # 1. Tokenize để lấy LABELS (Đáp án chuẩn: A cat...)
            tokenized = self.caption_tokenizer(
                caption_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            
            # Đây là Ground Truth (Đáp án): [A, cat, ..., EOS, PAD]
            labels = tokenized["input_ids"].squeeze()
            
            # 2. Tạo DECODER INPUT (Đầu vào: PAD A cat...)
            # Kỹ thuật: Shift Right (Dịch phải)
            # Lấy token PAD làm Start Token
            start_token_id = self.caption_tokenizer.pad_token_id
            
            # Tạo tensor chứa Start Token
            start_token_tensor = torch.tensor([start_token_id])
            
            # Input cho Decoder = [START] + [Labels bỏ token cuối]
            # Ví dụ Labels: [A, B, C, EOS] -> Decoder Input: [START, A, B, C]
            decoder_input_ids = torch.cat([start_token_tensor, labels[:-1]])
            
            # Tạo Attention Mask cho Decoder Input (Để model biết START là token thật)
            decoder_attention_mask = torch.cat([torch.tensor([1]), tokenized["attention_mask"].squeeze()[:-1]])

            return {
                "image": img,
                "decoder_input_ids": decoder_input_ids, # <-- Input đưa vào model (có START)
                "labels": labels,                       # <-- Target để tính Loss (ko có START)
                "attention_mask": decoder_attention_mask,
                "raw_text": caption_text
            }
        else:
            return {"image": img, "caption": caption_text}

class SampleCaption(nn.Module):
    """Chọn ngẫu nhiên 1 caption (và ép về string an toàn)."""
    def __call__(self, sample):
        if isinstance(sample, list) and len(sample) > 0:
            cap = random.choice(sample)
            if isinstance(cap, list):  # Trường hợp lồng
                cap = cap[0]
            return str(cap)
        return str(sample)
