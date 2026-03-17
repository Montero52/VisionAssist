import torch
import numpy as np
import torch.nn.functional as F
import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from collections import deque

class DepthEstimator:
    def __init__(self, device, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        self.device = device
        self.history = deque(maxlen=5) 
        
        print(f">> [Distance] Đang tải model: {model_name}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print("[Distance] Load model thành công!")
        except Exception as e:
            print(f"[Distance] Lỗi load model: {e}")
            raise e

    def estimate_distance(self, cv_image):
        """
        Input: Ảnh Numpy
        Output: Tuple (Khoảng cách mét, Vị trí vật, Tọa độ hộp đỏ)
        """
        # 1. Chạy Model
        inputs = self.processor(images=cv_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            output = self.model(pixel_values=pixel_values)
            predicted_depth = output.predicted_depth
            prediction = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=cv_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        h, w = depth_map.shape
        
        # ====================================================
        # BƯỚC MỚI: TỰ ĐỘNG TÌM VẬT GẦN NHẤT (AUTO-SCAN)
        # ====================================================
        
        # 1. Tạo mặt nạ (Mask) để AI chỉ tập trung vào vùng an toàn
        # Bỏ 20% trên cùng (Trần nhà/Bầu trời) -> Không quan tâm
        # Bỏ 10% dưới cùng (Mặt đất ngay chân) -> Tránh báo sai
        mask = np.zeros_like(depth_map, dtype=np.uint8)
        mask[int(h*0.2):int(h*0.9), :] = 255 
        
        # 2. Tìm điểm sáng nhất trong vùng Mask (Điểm gần nhất)
        # minMaxLoc trả về vị trí của giá trị nhỏ nhất và lớn nhất
        # Trong DepthAnything: Giá trị LỚN (Max) = GẦN
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(depth_map, mask=mask)
        
        # max_loc chính là tọa độ (x, y) của vật gần nhất!
        closest_x, closest_y = max_loc
        
        # 3. Tạo ROI động xung quanh điểm đó
        roi_size = 60 # Kích thước khung đo (pixel)
        x1 = max(0, closest_x - roi_size // 2)
        y1 = max(0, closest_y - roi_size // 2)
        x2 = min(w, closest_x + roi_size // 2)
        y2 = min(h, closest_y + roi_size // 2)
        
        # Cập nhật lại center thực tế để tính toán hình học
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Lấy giá trị đo
        center_roi = depth_map[y1:y2, x1:x2]
        if center_roi.size == 0: return 5.0, "An toàn", (0,0,0,0)
        
        #depth_val = np.median(center_roi)
        depth_val = np.percentile(center_roi, 90)
        if depth_val <= 0: return 5.0, "An toàn", (0,0,0,0)

        # ====================================================
        # BƯỚC 1: TÍNH Z (HỒI QUY ĐA THỨC)
        # ====================================================
        A, B, C = -0.13094, 0.33910, 1.56904
        Z_meters = (A * (depth_val ** 2)) + (B * depth_val) + C
        
        if Z_meters < 0.3: Z_meters = 0.3
        if Z_meters > 5.0: Z_meters = 5.0

        # ====================================================
        # BƯỚC 2: TÍNH RHO (HÌNH HỌC CẦU - SPHERICAL)
        # ====================================================
        # Quan trọng: Dùng center_x, center_y ĐỘNG vừa tìm được
        
        FOV_DEG = 60.0
        f_pixel = (w / 2) / np.tan(np.deg2rad(FOV_DEG / 2))
        
        x_norm = (center_x - (w / 2)) / f_pixel
        y_norm = (center_y - (h / 2)) / f_pixel
        
        rho_meters = Z_meters * np.sqrt(1 + x_norm**2 + y_norm**2)

        # ====================================================
        # XÁC ĐỊNH VỊ TRÍ (TRÁI / PHẢI / GIỮA)
        # ====================================================
        position_text = "Phía trước"
        if center_x < w * 0.33:
            position_text = "Bên Trái"
        elif center_x > w * 0.66:
            position_text = "Bên Phải"

        # Làm mượt
        self.history.append(rho_meters)
        smooth_dist = sum(self.history) / len(self.history)
        
        # Trả về thêm tọa độ box để vẽ lên màn hình
        return round(smooth_dist, 2), position_text, (x1, y1, x2, y2)