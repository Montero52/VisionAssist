import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import random

class JsonCaptionsDataset(Dataset):
    def __init__(self, root, annFile, image_transform=None, caption_tokenizer=None, 
                 max_len=64, img_key="file_name", cap_key="captions"):
        self.root = root
        self.image_transform = image_transform
        self.caption_tokenizer = caption_tokenizer
        self.max_len = max_len
        self.img_key = img_key
        self.cap_key = cap_key

        # 1. Đọc file JSON
        try:
            with open(annFile, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Lỗi không đọc được file JSON: {annFile}\nChi tiết: {e}")

        # 2. Xử lý format
        records = data["images"] if isinstance(data, dict) and "images" in data else data

        self.items = []
        for d in records:
            # Lấy tên file ảnh
            fn = d.get(self.img_key)
            # Lấy list caption
            caps = d.get(self.cap_key, [])
            
            # Kiểm tra dữ liệu
            if not fn or not caps:
                continue
                
            # Đảm bảo caption là string
            caps = [str(c).strip() for c in caps if isinstance(c, (str, list))]
            
            if len(caps) > 0:
                # Lưu đường dẫn đầy đủ và caption
                # fn có thể là '1000268201.jpg'
                self.items.append({
                    "image_path": os.path.join(root, fn),
                    "captions": caps,
                    "image_id": fn # Lưu ID để debug
                })

        print(f"Đã tải Dataset: {len(self.items)} ảnh.")
        if not self.items:
            raise RuntimeError("Dataset trống! Kiểm tra lại đường dẫn ảnh hoặc file JSON.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = item["image_path"]
        image_id = item["image_id"]
        caps = item["captions"]

        # --- A. XỬ LÝ ẢNH ---
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"LỖI ẢNH HỎNG: {img_path} - {e}")
            img = Image.new('RGB', (224, 224), color='black')

        if self.image_transform:
            img = self.image_transform(img)

        # --- B. XỬ LÝ CAPTION ---
        caption_text = random.choice(caps)

        # Tokenize cơ bản (Không Shift, không thêm START ở đây)
        # Để train.py tự xử lý việc cắt input/target cho đồng bộ
        if self.caption_tokenizer:
            tokenized = self.caption_tokenizer(
                caption_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt"
            )
            
            # Đây là chuỗi token đầy đủ: [A, B, C, EOS, PAD...]
            input_ids = tokenized["input_ids"].squeeze()
            attention_mask = tokenized["attention_mask"].squeeze()

            return {
                "image": img,
                # Trả về input_ids gốc, train.py sẽ tự cắt [:-1] làm input và [1:] làm target
                "decoder_input_ids": input_ids, 
                "attention_mask": attention_mask,
                "raw_text": caption_text,
                "image_id": image_id  # <--- QUAN TRỌNG: Để in ra tên ảnh
            }
        else:
            return {"image": img, "caption": caption_text, "image_id": image_id}