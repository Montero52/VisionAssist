# src/main/distance.py
import torch
import numpy as np
import torch.nn.functional as F

class DistanceCalculator:
    def __init__(self, midas_model, transform, device, scale_factor=1000.0):
        self.midas = midas_model
        self.transform = transform
        self.device = device
        # Chỉ giữ lại đúng 1 thông số quan trọng nhất
        self.scale_factor = scale_factor 

    def estimate_distance(self, cv_image):
        """
        Chỉ trả về Khoảng cách Z (mét)
        """
        # 1. Tiền xử lý & Chạy Model
        input_batch = self.transform(cv_image).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=cv_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        h, w = depth_map.shape
        
        # 2. Lấy vùng trung tâm (20% giữa ảnh)
        # Đây là vùng "ROI cố định" mà chúng ta đã thống nhất
        crop_ratio = 0.2
        center_h, center_w = int(h * crop_ratio), int(w * crop_ratio)
        y1, y2 = (h - center_h) // 2, (h - center_h) // 2 + center_h
        x1, x2 = (w - center_w) // 2, (w - center_w) // 2 + center_w
        
        center_roi = depth_map[y1:y2, x1:x2]
        
        # 3. Tính toán khoảng cách
        if center_roi.size == 0: return 0.0
            
        depth_val = np.median(center_roi)
        
        if depth_val < 0.1: return 99.9 # Quá xa hoặc lỗi
        
        # Công thức đơn giản nhất: Z = Scale / Depth
        Z_meters = self.scale_factor / depth_val
        
        return Z_meters