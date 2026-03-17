import os
import sys
import time
import json
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import T5Tokenizer

from torch.utils.data import Subset
from nltk.translate.meteor_score import meteor_score
from torchmetrics.text import BLEUScore

# Import config và modules
# Lấy đường dẫn tuyệt đối của thư mục chứa file train.py (tức là folder scripts)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Lấy thư mục cha (thư mục gốc VisionAssist)
root_dir = os.path.dirname(current_dir)
# Thêm thư mục gốc vào hệ thống tìm kiếm của Python
sys.path.append(root_dir)

from config import vit_cfg, trans_cfg, epochs, image_dir, caption_dir, BASE_DIR
from src.main.model.model import ViT_Transformer
from src.data.dataset import JsonCaptionsDataset
from src.utils.utils import save_checkpoint

# --- [AMP] Import thư viện Mixed Precision ---
from torch.cuda.amp import GradScaler, autocast

# --- HÀM TRAIN ---
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # T5 Start Token = 0
    start_token_id = 0 
    
    print(f"\n[INFO] Bat dau Epoch {epoch+1}...")
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        
        full_tokens = batch['decoder_input_ids'].to(device) 
        full_mask   = batch['attention_mask'].to(device)
        
        # 1. Tao Targets (Thay 0 bang -100 de bo qua khi tinh Loss)
        targets = full_tokens.clone()
        targets[targets == 0] = -100 
        
        # 2. Tao Decoder Input (Them Start Token vao dau)
        batch_size = full_tokens.size(0)
        start_col = torch.full((batch_size, 1), start_token_id, device=device, dtype=torch.long)
        decoder_input = torch.cat([start_col, full_tokens[:, :-1]], dim=1)
        
        # 3. Tao Mask cho Decoder Input
        mask_col = torch.full((batch_size, 1), 1, device=device, dtype=torch.long)
        attention_mask = torch.cat([mask_col, full_mask[:, :-1]], dim=1)
        
        optimizer.zero_grad()

        # [AMP] Forward
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(images, decoder_input, padding_mask=attention_mask)
            output_dim = outputs.shape[-1]
            loss = criterion(outputs.reshape(-1, output_dim), targets.reshape(-1))
        
        # [AMP] Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # In log moi 100 batch de tranh spam man hinh
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    end_time = time.time()
    print(f"=== Ket thuc Epoch {epoch+1} | Loss TB: {avg_loss:.4f} | Thoi gian: {end_time - start_time:.2f}s ===")
    return avg_loss

# --- TAI NLTK ---
try:
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/omw-1.4.zip')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("[INFO] Dang tai du lieu NLTK...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('punkt_tab')

# --- HÀM EVALUATE ---
def evaluate_model(model, dataloader, criterion, tokenizer, device):
    model.eval()
    
    # Bien luu Val Loss
    total_val_loss = 0
    
    bleu4_metric = BLEUScore(n_gram=4)
    preds_str = []
    targets_str = []
    meteor_scores = []
    
    # Token dac biet cho tinh Loss
    start_token_id = 0 
    
    print("\n[INFO] Dang chay danh gia (Evaluation & Calc Loss)...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            raw_captions = batch['raw_text'] 
            
            # --- PHAN 1: TINH VAL LOSS ---
            full_tokens = batch['decoder_input_ids'].to(device) 
            full_mask   = batch['attention_mask'].to(device)
            
            targets = full_tokens.clone()
            targets[targets == 0] = -100 
            
            batch_size = full_tokens.size(0)
            start_col = torch.full((batch_size, 1), start_token_id, device=device, dtype=torch.long)
            decoder_input = torch.cat([start_col, full_tokens[:, :-1]], dim=1)
            
            mask_col = torch.full((batch_size, 1), 1, device=device, dtype=torch.long)
            attention_mask = torch.cat([mask_col, full_mask[:, :-1]], dim=1)
            
            # Forward tinh Loss
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images, decoder_input, padding_mask=attention_mask)
                output_dim = outputs.shape[-1]
                loss = criterion(outputs.reshape(-1, output_dim), targets.reshape(-1))
            
            total_val_loss += loss.item()

            # --- PHAN 2: SINH CAPTION & TINH DIEM ---
            if batch_idx == 0:
                print(f"\n--- KET QUA MAU TAI BATCH 0 ---")
                num_show = min(len(images), 3) 
                for i in range(num_show):
                    img_tensor = images[i].unsqueeze(0)
                    generated_cap = model.beam_search(img_tensor, tokenizer, beam_size=3, max_len=30, device=device)

                    print(f" Anh ID: {batch['image_id'][i] if 'image_id' in batch else 'Unknown'}")
                    print(f" Model:   '{generated_cap}'")
                    print(f" Dap an:  '{raw_captions[i]}'")
                    print("-------------------------------------------------")
                    
                    preds_str.append(generated_cap)
                    targets_str.append([raw_captions[i]])
                    
                    reference = raw_captions[i].lower()
                    reference_tokens = [nltk.word_tokenize(reference)]
                    hypothesis = generated_cap.lower()
                    hypothesis_tokens = nltk.word_tokenize(hypothesis)
                    score = meteor_score(reference_tokens, hypothesis_tokens)
                    meteor_scores.append(score)
            else:
                for i in range(len(images)):
                    img_tensor = images[i].unsqueeze(0)
                    generated_cap = model.beam_search(img_tensor, tokenizer, beam_size=3, max_len=30, device=device)
                    preds_str.append(generated_cap)
                    targets_str.append([raw_captions[i]])
                    
                    reference = raw_captions[i].lower()
                    reference_tokens = [nltk.word_tokenize(reference)]
                    hypothesis = generated_cap.lower()
                    hypothesis_tokens = nltk.word_tokenize(hypothesis)
                    score = meteor_score(reference_tokens, hypothesis_tokens)
                    meteor_scores.append(score)

            if batch_idx % 20 == 0:
                print(f"   Dang danh gia... {batch_idx}/{len(dataloader)}")

    avg_val_loss = total_val_loss / len(dataloader)
    score_bleu4 = bleu4_metric(preds_str, targets_str).item()
    score_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    
    return avg_val_loss, score_bleu4, score_meteor

# --- MAIN ---
def main():
    # 1. Cau hinh thiet bi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dang su dung thiet bi: {device}")
    
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    history_path = os.path.join(checkpoint_dir, "history.json")

    # 2. Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
    
    real_vocab_size = len(tokenizer)
    print(f"[INFO] T5 Vocab Size: {real_vocab_size}")
    trans_cfg['vocab_size'] = real_vocab_size 

    # 3. Khoi tao Model
    print("[INFO] Dang khoi tao Model...")
    model = ViT_Transformer(vit_cfg, trans_cfg, vocab_size=real_vocab_size).to(device)
    
    # 4. Transform
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),               
        transforms.RandomCrop((224, 224)),           
        transforms.RandomHorizontalFlip(p=0.5),      
        transforms.ColorJitter(brightness=0.1, contrast=0.1), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 5. Dataset & DataLoader
    train_dataset = JsonCaptionsDataset(
        root=image_dir,
        annFile=os.path.join(caption_dir, "train_captions.json"), 
        image_transform=train_transforms,
        caption_tokenizer=tokenizer,
        max_len=trans_cfg['max_len']
    )
    
    val_dataset = JsonCaptionsDataset(
        root=image_dir,
        annFile=os.path.join(caption_dir, "val_captions.json"), 
        image_transform=val_transforms,
        caption_tokenizer=tokenizer,
        max_len=trans_cfg['max_len']
    )

    print(f"[INFO] DANG CHAY CHE DO FULL DATASET ({len(train_dataset)} anh)")
    
    BATCH_SIZE = 16 
    num_workers = 2
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # 6. Unfreeze & Optimizer
    print("[INFO] CAU HINH: UNFREEZE ALL")
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.0)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # 7. Resume
    start_epoch = 0
    best_bleu = 0.0
    history = {"train_loss": [], "val_loss": [], "val_bleu": [], "val_meteor": [], "epochs": []}
    resume_path = os.path.join(checkpoint_dir, "last_model.pth")

    if os.path.exists(resume_path):
        print(f"[INFO] Phat hien checkpoint cu tai {resume_path}. Dang khoi phuc...")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        best_bleu = checkpoint.get('best_bleu', 0.0)
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        print(f"[INFO] Da khoi phuc! Tiep tuc train tu Epoch {start_epoch+1}. Best BLEU: {best_bleu:.4f}")
    else:
        print("[INFO] Khong tim thay checkpoint cu. Bat dau train moi.")

    # 8. Loop
    print(f">>> BAT DAU HUAN LUYEN TREN {len(train_dataset)} ANH (EPOCHS: {epochs}) <<<")
    
    for epoch in range(start_epoch, epochs):
        # A. Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler)
        
        # B. Evaluate
        val_loss, bleu4, meteor = 0.0, 0.0, 0.0

        if (epoch + 1) % 1 == 0:
            val_loss, bleu4, meteor = evaluate_model(model, val_loader, criterion, tokenizer, device)
            
            print(f"Ket qua Epoch {epoch+1}:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}") 
            print(f"   BLEU-4:     {bleu4:.4f}")
            print(f"   METEOR:     {meteor:.4f}")

            scheduler.step(val_loss)

            if bleu4 > best_bleu:
                best_bleu = bleu4
                best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
                save_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': train_loss,
                    'best_bleu': best_bleu
                }
                torch.save(save_data, best_ckpt_path)
                print(f"Model Tot Nhat Moi! (BLEU: {best_bleu:.4f}) -> {best_ckpt_path}")

            last_ckpt_path = os.path.join(checkpoint_dir, "last_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': train_loss,
                'best_bleu': best_bleu
            }, last_ckpt_path)
            print(f"   Da luu checkpoint dinh ky tai Epoch {epoch+1}")

        # D. Log
        history["epochs"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bleu"].append(bleu4)
        history["val_meteor"].append(meteor)
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Da cap nhat log vao: {history_path}")

        # E. Backup
        if (epoch + 1) % 1 == 0:
            backup_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), backup_path)

    print("HOAN THANH HUAN LUYEN!")

# --- QUAN TRỌNG: CHẮC CHẮN DÒNG NÀY PHẢI SÁT LỀ TRÁI ---
if __name__ == "__main__":
    main()