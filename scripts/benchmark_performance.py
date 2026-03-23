import time
import torch
import psutil
import os
import gc
import logging
import numpy as np
from pathlib import Path

# Thêm đường dẫn gốc để import modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main.model.model import ViT_Transformer
from src.main.model.openvino_engine import OpenVINOEngine
import config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("Benchmark")

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark():
    # Thông số từ config của bạn
    embed_dim = config.vit_cfg.get("embed_dim", 768)
    max_len = 32 # Chúng ta test với độ dài chuẩn 32 tokens
    
    dummy_img = torch.randn(1, 3, 224, 224)
    dummy_input_ids = np.zeros((1, max_len), dtype=np.int64)
    dummy_mask = np.zeros((1, 1, max_len, max_len), dtype=np.float32)
    
    results = {}

    # --- TEST 1: PYTORCH (BASE) ---
    logger.info("--- Starting PyTorch Benchmark ---")
    ram_start_pt = get_ram_usage()
    
    try:
        pt_model = ViT_Transformer(vit_config=config.vit_cfg, trans_cfg=config.trans_cfg)
        pt_model.eval()
        
        times_pt = []
        with torch.no_grad():
            for i in range(5):
                start = time.perf_counter()
                # Giả lập forward pass
                _ = pt_model(dummy_img, torch.ones((1, max_len), dtype=torch.long))
                times_pt.append(time.perf_counter() - start)
                if i == 0: logger.info("PyTorch Warm-up done.")
        
        results['PyTorch'] = {
            'avg_time': np.mean(times_pt[1:]), # Bỏ lần đầu (warm-up)
            'ram_peak': get_ram_usage() - ram_start_pt
        }
    except Exception as e:
        logger.error(f"PyTorch Test failed: {e}")
    
    del pt_model
    gc.collect()

    # --- TEST 2: OPENVINO (OPTIMIZED) ---
    logger.info("--- Starting OpenVINO Benchmark ---")
    ram_start_ov = get_ram_usage()
    
    try:
        ov_engine = OpenVINOEngine() 
        
        times_ov = []
        for i in range(5):
            start = time.perf_counter()
            # Sử dụng các hàm infer chính thức từ engine của bạn
            enc_out = ov_engine.infer_encoder(dummy_img)
            _ = ov_engine.infer_decoder(dummy_input_ids, enc_out, dummy_mask)
            times_ov.append(time.perf_counter() - start)
            if i == 0: logger.info("OpenVINO Warm-up done.")

        results['OpenVINO'] = {
            'avg_time': np.mean(times_ov[1:]),
            'ram_peak': get_ram_usage() - ram_start_ov
        }
    except Exception as e:
        logger.error(f"OpenVINO Test failed: {e}")

    # --- REPORTING ---
    if 'PyTorch' in results and 'OpenVINO' in results:
        print("\n" + "="*60)
        print(f"{'METRIC':<25} | {'PYTORCH':<15} | {'OPENVINO':<15}")
        print("-" * 60)
        
        t_pt = results['PyTorch']['avg_time']
        t_ov = results['OpenVINO']['avg_time']
        r_pt = results['PyTorch']['ram_peak']
        r_ov = results['OpenVINO']['ram_peak']
        
        print(f"{'Avg Inference (s)':<25} | {t_pt:>14.4f} | {t_ov:>14.4f}")
        print(f"{'Peak RAM Usage (MB)':<25} | {r_pt:>14.1f} | {r_ov:>14.1f}")
        print("="*60)
        print(f"Speedup: {t_pt / t_ov:.2f}x faster")
        print(f"RAM Saving: {r_pt - r_ov:.1f} MB")
        print("="*60)

if __name__ == "__main__":
    benchmark()