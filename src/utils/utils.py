import os
import torch

CHECKPOINT_DIR = "checkpoints"   # đổi nếu muốn
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Thêm: lưu theo chu kỳ bao nhiêu epoch 1 lần
SAVE_EVERY = 50  # lưu mỗi 50 epoch

def save_checkpoint(model, optimizer, scaler, epoch, step, path):
    """Lưu checkpoint gọn (model, optimizer, scaler, epoch, step)."""
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
    try:
        tokenizer.save_pretrained(out_dir)
    except Exception as e:
        print(f"[WARN] Không thể lưu tokenizer: {e}")

def load_checkpoint(model, optimizer, scaler, path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler is not None and ckpt.get("scaler_state") is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    start_epoch = ckpt.get("epoch", 0) + 1
    start_step = ckpt.get("step", 0)
    return start_epoch, start_step