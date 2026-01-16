import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import T5Tokenizer
import os
import time

import nltk
from nltk.translate.meteor_score import meteor_score
from torchmetrics.text import BLEUScore

# Import từ các module của bạn
# Đảm bảo bạn đã có file config.py và cấu trúc thư mục đúng
from config import vit_cfg, trans_cfg, epochs, image_dir, caption_dir, BASE_DIR
from src.main.model.model import ViT_Transformer
from src.data.dataset import JsonCaptionsDataset
from src.utils.utils import save_checkpoint

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    print(f"\n>>> Bắt đầu Epoch {epoch+1}...")
    
    for batch_idx, batch in enumerate(dataloader):
        # 1. Lấy dữ liệu từ dataloader (đã được xử lý trong Dataset)
        images = batch['image'].to(device)
        
        # decoder_input_ids: Đã có PAD ở đầu (Input cho model)
        decoder_input_ids = batch['decoder_input_ids'].to(device) 
        
        # labels: Đã là từ thật (Target để tính loss)
        labels = batch['labels'].to(device)
        
        # attention_mask: Mask cho decoder_input_ids
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()

        # 2. Forward Pass
        # Không cần cắt gọt gì nữa, nạp thẳng vào model
        outputs = model(images, decoder_input_ids, padding_mask=attention_mask) # [B, T, Vocab]
        
        # 3. Tính Loss
        # Reshape lại để khớp với CrossEntropyLoss: (N, C) vs (N)
        # outputs: [Batch, Length, Vocab] -> [Batch*Length, Vocab]
        # labels:  [Batch, Length]        -> [Batch*Length]
        
        output_dim = outputs.shape[-1]
        loss = criterion(outputs.view(-1, output_dim), labels.view(-1))
        
        # 4. Backward & Optimize
        loss.backward()
        
        # Gradient Clipping (Chống bùng nổ gradient)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()

        # Log tiến độ
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    end_time = time.time()
    print(f"=== Kết thúc Epoch {epoch+1} | Loss TB: {avg_loss:.4f} | Thời gian: {end_time - start_time:.2f}s ===")
    return avg_loss

try:
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/omw-1.4.zip')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print(">> Đang tải dữ liệu NLTK...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('punkt_tab')

def evaluate_model(model, dataloader, tokenizer, device):
    """
    Đánh giá model sử dụng:
    1. BLEU-4 (Torchmetrics)
    2. METEOR (NLTK thuần)
    """
    model.eval()
    
    # BLEU nhận input là chuỗi string, tự tách từ bên trong
    bleu4_metric = BLEUScore(n_gram=4)
    
    preds_str = []   # Cho BLEU
    targets_str = [] # Cho BLEU
    meteor_scores = [] # Cho METEOR
    
    print("\n>>> Đang chạy đánh giá (Evaluation)...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            
            # Lấy caption gốc (raw text) từ dataset
            # Lưu ý: raw_captions là tuple hoặc list các chuỗi
            raw_captions = batch['raw_text'] 

            for i in range(len(images)):
                img_tensor = images[i].unsqueeze(0)
                
                # 1. Model sinh caption
                # Gọi hàm sinh caption (Autoregressive)
                # Max len ngắn thôi (30) để chạy cho nhanh
                # generated_cap = model.generate(
                #     img_tensor, 
                #     tokenizer, 
                #     max_len=30, 
                #     device=device,
                #     top_k=5, 
                #     top_p=0.9
                # )
                generated_cap = model.beam_search(
                    img_tensor, 
                    tokenizer, 
                    beam_size=3, # Beam nhỏ (3) để chạy nhanh trên CPU
                    max_len=30, 
                    device=device
                )
                
                # --- CHUẨN BỊ DỮ LIỆU ---
                
                # A. Cho BLEU (String)
                preds_str.append(generated_cap)
                targets_str.append([raw_captions[i]]) # List of List
                
                # B. Cho METEOR (Tokenized List)
                # Quan trọng: METEOR của NLTK yêu cầu list các từ (tokens)
                # Và nên đưa về chữ thường (.lower()) để so sánh chính xác hơn
                
                # Đáp án (Reference): Phải là list các câu, mỗi câu là list từ
                # Ví dụ: [['a', 'cat', 'sitting']]
                reference = raw_captions[i].lower()
                reference_tokens = [nltk.word_tokenize(reference)]
                
                # Dự đoán (Hypothesis): List từ
                # Ví dụ: ['a', 'cat', 'sit']
                hypothesis = generated_cap.lower()
                hypothesis_tokens = nltk.word_tokenize(hypothesis)
                
                # Tính điểm METEOR cho cặp này
                score = meteor_score(reference_tokens, hypothesis_tokens)
                meteor_scores.append(score)
            
            if batch_idx % 20 == 0:
                print(f"   Batch {batch_idx}/{len(dataloader)}...")

    # Tính điểm trung bình
    score_bleu4 = bleu4_metric(preds_str, targets_str).item()
    
    # Trung bình METEOR
    score_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    
    return score_bleu4, score_meteor

def main():
    # 1. Cấu hình thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")
    
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 2. Tokenizer (Giữ nguyên như code trước)
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    # ... (Logic kiểm tra tokenizer giữ nguyên) ...

    # 3. Transform (Giữ nguyên)
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),               
        transforms.RandomCrop((224, 224)),           
        transforms.RandomHorizontalFlip(p=0.5),      
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # Val chỉ resize, không crop/flip ngẫu nhiên
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # 4. Dataset & DataLoader
    # --- TẬP TRAIN ---
    train_dataset = JsonCaptionsDataset(
        root=image_dir,
        # Trỏ vào file json dành riêng cho train
        annFile=os.path.join(caption_dir, "train_captions.json"), 
        image_transform=train_transforms,
        caption_tokenizer=tokenizer,
        max_len=trans_cfg['max_len']
    )
    
    # --- TẬP VAL ---
    val_dataset = JsonCaptionsDataset(
        root=image_dir,
        # Trỏ vào file json dành riêng cho val
        annFile=os.path.join(caption_dir, "val_captions.json"), 
        image_transform=val_transforms,
        caption_tokenizer=tokenizer,
        max_len=trans_cfg['max_len']
    )
    
    # Chia nhỏ dataset để demo (nếu máy yếu)
    # train_subset, val_subset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.9), int(len(train_dataset)*0.1)])

    BATCH_SIZE = 16
    num_workers = 0 if os.name == 'nt' else 2
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    
    # Val loader không cần shuffle, batch_size nhỏ để chạy evaluate cho nhẹ
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=num_workers)

    # 5. Model Setup (Giữ nguyên)
    print("Đang khởi tạo Model...")
    model = ViT_Transformer(vit_cfg, trans_cfg, vocab_size=len(tokenizer)).to(device)

    # 6. Loss & Optimizer (Giữ nguyên)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # --- BIẾN ĐỂ LƯU BEST MODEL ---
    best_bleu = 0.0

    # 7. Training Loop
    print(f">>> BẮT ĐẦU HUẤN LUYỆN TRÊN {len(train_dataset)} ẢNH <<<")
    
    for epoch in range(epochs):
        # A. Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        scheduler.step(train_loss)
        
        # B. Evaluate (Chạy mỗi 2 epoch hoặc epoch cuối để tiết kiệm thời gian)
        if (epoch + 1) % 2 == 0 or (epoch + 1) == epochs:
            bleu4, meteor = evaluate_model(model, val_loader, tokenizer, device)
            
            print(f"Kết quả đánh giá Epoch {epoch+1}:")
            print(f"   BLEU-4: {bleu4:.4f}")
            print(f"   METEOR: {meteor:.4f}")
            
            # C. Lưu Best Model (Quan trọng hơn lưu theo epoch)
            if bleu4 > best_bleu:
                best_bleu = bleu4
                best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
                try:
                    save_checkpoint(model, optimizer, scheduler, epoch, train_loss, best_ckpt_path)
                except:
                    torch.save(model.state_dict(), best_ckpt_path)
                print(f"Tìm thấy Model tốt mới! (BLEU: {best_bleu:.4f}). Đã lưu tại: {best_ckpt_path}")

        # Lưu checkpoint định kỳ (Backup)
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)

    print("HOÀN THÀNH HUẤN LUYỆN!")

if __name__ == "__main__":
    main()