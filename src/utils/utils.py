import os
import torch
import logging

logger = logging.getLogger("Utils")

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Set: Save checkpoint every N epochs
SAVE_EVERY = 50 

def save_checkpoint(model, optimizer, scaler, epoch, step, path):
    """Saves a concise checkpoint (model, optimizer, scaler, epoch, step)."""
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "torch_version": torch.__version__,
    }
    torch.save(ckpt, path)

def save_tokenizer(tokenizer, out_dir):
    """Saves the tokenizer files."""
    try:
        tokenizer.save_pretrained(out_dir)
    except Exception as e:
        logger.warning(f"Could not save tokenizer: {e}")

def load_checkpoint(model, optimizer, scaler, path, map_location="cpu"):
    """Loads checkpoint state dictionaries and returns starting epoch/step."""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
        
    start_epoch = ckpt.get("epoch", 0) + 1
    start_step = ckpt.get("step", 0)
    
    return start_epoch, start_step
