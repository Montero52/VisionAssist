import sys
import os
import logging
import cv2
import numpy as np
from PIL import Image # Cần import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
import time
from src.main.model.openvino_engine import OpenVINOEngine
from transformers import T5Tokenizer
# THÊM IMPORT IMAGE PROCESSOR CHO DEPTH MODEL
from transformers import AutoImageProcessor
from torchvision import transforms


logger = logging.getLogger("EdgeCaseTester")

def create_dummy_frame(h=480, w=640, mode="normal"):
    """Creates a dummy image frame simulating different conditions."""
    if mode == "low_light":
        base = np.random.randint(0, 30, size=(h, w, 3), dtype=np.uint8)
        cv2.circle(base, (w//2, h//2), 50, (100, 100, 100), -1)
        return base
    elif mode == "partial_occlusion":
        base = np.random.randint(50, 150, size=(h, w, 3), dtype=np.uint8)
        base[int(h*0.6):, int(w*0.6):, :] = 0 
        return base
    return np.random.randint(100, 255, size=(h, w, 3), dtype=np.uint8)

def run_edge_case_test(ov_engine, tokenizer, depth_processor, mode: str):
    """Runs inference under a specific edge case mode and checks for NaN/Crash."""
    logger.info(f"--- Running Edge Case Test: {mode.upper()} ---")
    
    h, w = 480, 640
    img_np = create_dummy_frame(h, w, mode=mode)

    # Convert Numpy to PIL for processing
    pil_image = Image.fromarray(img_np)

    # 1. Test Depth Estimation
    try:
        # SỬA LỖI Ở ĐÂY: Tiền xử lý ảnh cho Depth Model
        inputs = depth_processor(images=pil_image, return_tensors="np")
        pixel_values = inputs.pixel_values
        
        # Đưa tensor đã tiền xử lý vào OpenVINO Engine
        # Lưu ý: Cần chắc chắn hàm infer_depth nhận numpy array đã resize đúng chuẩn
        depth_map = ov_engine.infer_depth(pixel_values) 
        
        if np.isnan(depth_map).any():
            logger.error(f"Depth Inference resulted in NaN values for mode: {mode}")
            return False
        logger.info(f"Depth Check OK. Shape: {depth_map.shape}")
    except Exception as e:
        logger.error(f"Depth Inference CRASHED for mode {mode}: {e}")
        return False

# 2. Test Captioning (Only run if mode is not "distance_only")
    if mode != "distance_only":
        try:
            # Tự định nghĩa Image Transform tại đây thay vì gọi từ config
            image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.685, 0.656, 0.606], std=[0.229, 0.224, 0.225])
            ])
            
            # Tiền xử lý ảnh cho Captioning Model
            img_tensor = image_transform(pil_image).unsqueeze(0).numpy()
            
            caption_en = ov_engine.beam_search(
                img_tensor,
                tokenizer,
                beam_size=1,
                max_len=20
            )
            
            if caption_en is None or not isinstance(caption_en, str) or caption_en.strip() == "":
                 logger.error(f"Captioning produced empty or invalid string for mode: {mode}")
                 return False
            
            logger.info(f"Caption Check OK. Output: '{caption_en[:30]}...'")
        except Exception as e:
            logger.error(f"Caption Inference CRASHED for mode {mode}: {e}")
            return False
            
    logger.info(f"Edge Case '{mode}' Test PASSED stability check.")
    return True

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    try:
        ov_engine = OpenVINOEngine() 
        tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
        # KHỞI TẠO IMAGE PROCESSOR CHO DEPTH MODEL (Giống như khi bạn dùng PyTorch)
        depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}. Cannot proceed with Edge Case Test.")
        exit(1)
        
    results = {
        "low_light": run_edge_case_test(ov_engine, tokenizer, depth_processor, "low_light"),
        "partial_occlusion": run_edge_case_test(ov_engine, tokenizer, depth_processor, "partial_occlusion"),
        "normal": run_edge_case_test(ov_engine, tokenizer, depth_processor, "normal")
    }
    
    print("\n======== SUMMARY ========")
    all_passed = True
    for test, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test.upper():<20}: {status}")
        if not passed:
            all_passed = False
    print("=========================")
    if all_passed:
        print("All stability tests passed successfully!")