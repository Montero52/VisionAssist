import os
import sys
import time
import json
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import T5Tokenizer

from torch.utils.data import Subset
from nltk.translate.meteor_score import meteor_score
from torchmetrics.text import BLEUScore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log", encoding='utf-8')
    ]
)
logger = logging.getLogger("Training")

# Import config and modules
# Get absolute path of the directory containing train.py (scripts folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get root directory (VisionAssist)
root_dir = os.path.dirname(current_dir)
# Add root directory to Python path
sys.path.append(root_dir)

from config import vit_cfg, trans_cfg, epochs, image_dir, caption_dir, BASE_DIR
from src.main.model.model import ViT_Transformer
from src.data.dataset import JsonCaptionsDataset
from src.utils.utils import save_checkpoint

# --- [AMP] Import Mixed Precision library ---
from torch.cuda.amp import GradScaler, autocast

# --- TRAINING FUNCTION ---
def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, scaler):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # T5 Start Token = 0
    start_token_id = 0 
    
    logger.info(f"Starting Epoch {epoch+1}...")
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        
        full_tokens = batch['decoder_input_ids'].to(device) 
        full_mask   = batch['attention_mask'].to(device)
        
        # 1. Create Targets (Replace 0 with -100 to ignore in Loss calculation)
        targets = full_tokens.clone()
        targets[targets == 0] = -100 
        
        # 2. Create Decoder Input (Prepend Start Token)
        batch_size = full_tokens.size(0)
        start_col = torch.full((batch_size, 1), start_token_id, device=device, dtype=torch.long)
        decoder_input = torch.cat([start_col, full_tokens[:, :-1]], dim=1)
        
        # 3. Create Mask for Decoder Input
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

        # Log every 100 batches to avoid flooding
        if batch_idx % 100 == 0:
            logger.info(f"Epoch [{epoch+1}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    end_time = time.time()
    logger.info(f"=== Finished Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Time: {end_time - start_time:.2f}s ===")
    return avg_loss

# --- DOWNLOAD NLTK DATA ---
try:
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('corpora/omw-1.4.zip')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.info("Downloading NLTK data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('punkt_tab')

# --- EVALUATE FUNCTION ---
def evaluate_model(model, dataloader, criterion, tokenizer, device):
    model.eval()
    
    # Store Val Loss
    total_val_loss = 0
    
    bleu4_metric = BLEUScore(n_gram=4)
    preds_str = []
    targets_str = []
    meteor_scores = []
    
    # Special token for loss calculation
    start_token_id = 0 
    
    logger.info("Running evaluation (Calc Loss & Metrics)...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            raw_captions = batch['raw_text'] 
            
            # --- PART 1: CALC VAL LOSS ---
            full_tokens = batch['decoder_input_ids'].to(device) 
            full_mask   = batch['attention_mask'].to(device)
            
            targets = full_tokens.clone()
            targets[targets == 0] = -100 
            
            batch_size = full_tokens.size(0)
            start_col = torch.full((batch_size, 1), start_token_id, device=device, dtype=torch.long)
            decoder_input = torch.cat([start_col, full_tokens[:, :-1]], dim=1)
            
            mask_col = torch.full((batch_size, 1), 1, device=device, dtype=torch.long)
            attention_mask = torch.cat([mask_col, full_mask[:, :-1]], dim=1)
            
            # Forward for Loss calculation
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images, decoder_input, padding_mask=attention_mask)
                output_dim = outputs.shape[-1]
                loss = criterion(outputs.reshape(-1, output_dim), targets.reshape(-1))
            
            total_val_loss += loss.item()

            # --- PART 2: GENERATE CAPTION & CALC SCORES ---
            if batch_idx == 0:
                logger.info("--- SAMPLE RESULTS AT BATCH 0 ---")
                num_show = min(len(images), 3) 
                for i in range(num_show):
                    img_tensor = images[i].unsqueeze(0)
                    generated_cap = model.beam_search(img_tensor, tokenizer, beam_size=3, max_len=30, device=device)

                    logger.info(f" Image ID: {batch['image_id'][i] if 'image_id' in batch else 'Unknown'}")
                    logger.info(f" Model:   '{generated_cap}'")
                    logger.info(f" Ground Truth:  '{raw_captions[i]}'")
                    logger.info("-------------------------------------------------")
                    
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
                logger.info(f"   Evaluating... {batch_idx}/{len(dataloader)}")

    avg_val_loss = total_val_loss / len(dataloader)
    score_bleu4 = bleu4_metric(preds_str, targets_str).item()
    score_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    
    return avg_val_loss, score_bleu4, score_meteor

# --- MAIN ---
def main():
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    history_path = os.path.join(checkpoint_dir, "history.json")

    # 2. Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
    
    real_vocab_size = len(tokenizer)
    logger.info(f"T5 Vocab Size: {real_vocab_size}")
    trans_cfg['vocab_size'] = real_vocab_size 

    # 3. Initialize Model
    logger.info("Initializing Model...")
    model = ViT_Transformer(vit_cfg, trans_cfg, vocab_size=real_vocab_size).to(device)
    
    # 4. Transforms
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

    logger.info(f"RUNNING IN FULL DATASET MODE ({len(train_dataset)} images)")
    
    BATCH_SIZE = 16 
    num_workers = 2
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # 6. Unfreeze & Optimizer
    logger.info("CONFIGURATION: UNFREEZE ALL")
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
        logger.info(f"Old checkpoint detected at {resume_path}. Restoring...")
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
        logger.info(f"Restored! Continuing training from Epoch {start_epoch+1}. Best BLEU: {best_bleu:.4f}")
    else:
        logger.info("No old checkpoint found. Starting fresh training.")

    # 8. Loop
    logger.info(f">>> STARTING TRAINING ON {len(train_dataset)} IMAGES (EPOCHS: {epochs}) <<<")
    
    for epoch in range(start_epoch, epochs):
        # A. Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler)
        
        # B. Evaluate
        val_loss, bleu4, meteor = 0.0, 0.0, 0.0

        if (epoch + 1) % 1 == 0:
            val_loss, bleu4, meteor = evaluate_model(model, val_loader, criterion, tokenizer, device)
            
            logger.info(f"Results Epoch {epoch+1}:")
            logger.info(f"   Train Loss: {train_loss:.4f}")
            logger.info(f"   Val Loss:   {val_loss:.4f}") 
            logger.info(f"   BLEU-4:     {bleu4:.4f}")
            logger.info(f"   METEOR:     {meteor:.4f}")

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
                logger.info(f"New Best Model! (BLEU: {best_bleu:.4f}) -> {best_ckpt_path}")

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
            logger.info(f"   Saved periodic checkpoint at Epoch {epoch+1}")

        # D. Log
        history["epochs"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_bleu"].append(bleu4)
        history["val_meteor"].append(meteor)
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        logger.info(f"Updated logs in: {history_path}")

        # E. Backup
        if (epoch + 1) % 1 == 0:
            backup_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), backup_path)

    logger.info("TRAINING COMPLETED!")

if __name__ == "__main__":
    main()
