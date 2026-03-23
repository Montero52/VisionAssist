import torch
import numpy as np
import torch.nn.functional as F
import cv2
import logging
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from collections import deque
import config
from src.main.model.openvino_engine import OpenVINOEngine

logger = logging.getLogger("DistanceEstimator")

class DepthEstimator:
    def __init__(self, device, model_name="depth-anything/Depth-Anything-V2-Small-hf", use_openvino: bool = False):
        """
        Initializes the Depth Estimator with a pretrained Depth-Anything-V2 model.
        Args:
            device (str): Device to run inference on (e.g., 'cpu', 'cuda', 'auto').
            model_name (str): Hugging Face model repository name.
            use_openvino (bool): Whether to use OpenVINO for inference.
        """
        self.device = device
        self.use_openvino = use_openvino
        self.history = deque(maxlen=5) 
        
        logger.info(f"Loading Depth Model: {model_name} (OpenVINO={self.use_openvino})...")
        
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
            
            if self.use_openvino:
                # Use OpenVINO Engine
                self.ov_engine = OpenVINOEngine()
                logger.info("Depth Model loaded via OpenVINO Engine!")
            else:
                # Use PyTorch
                self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
                self.model.eval()
                logger.info("Depth Model loaded via PyTorch successfully!")
                
        except Exception as e:
            logger.error(f"Error loading depth model: {e}")
            raise e

    def estimate_distance(self, cv_image: np.ndarray):
        """
        Estimates the distance to the closest object in the center/safe region of the frame.
        Input: Numpy array image (RGB)
        Output: Tuple (Distance in meters, Position text, Red box coordinates)
        """
        # 1. Model Inference
        inputs = self.processor(images=cv_image, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        
        if self.use_openvino:
            # OpenVINO Inference
            depth_map = self.ov_engine.infer_depth(pixel_values)
            # OpenVINO returns [1, H, W], we need to squeeze it
            if depth_map.ndim == 3:
                depth_map = depth_map.squeeze(0)
            
            # Interpolate to match image size if necessary
            if depth_map.shape != cv_image.shape[:2]:
                depth_map = cv2.resize(depth_map, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:
            # PyTorch Inference
            pixel_values = pixel_values.to(self.device)
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
        # AUTO-SCAN: LOCATE THE CLOSEST OBJECT
        # ====================================================
        
        # 1. Create a mask to focus on the safe region
        # Ignore top 20% (ceiling/sky) and bottom 10% (ground near feet)
        mask = np.zeros_like(depth_map, dtype=np.uint8)
        mask[int(h*0.2):int(h*0.9), :] = 255 
        
        # 2. Find the brightest point in the masked depth map (closest point)
        # In DepthAnything: High values = Closer
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(depth_map, mask=mask)
        
        # max_loc is the (x, y) coordinates of the closest object!
        closest_x, closest_y = max_loc
        
        # 3. Create a dynamic ROI (Region of Interest) around the point
        roi_size = 60 # Measurement frame size (pixels)
        x1 = max(0, closest_x - roi_size // 2)
        y1 = max(0, closest_y - roi_size // 2)
        x2 = min(w, closest_x + roi_size // 2)
        y2 = min(h, closest_y + roi_size // 2)
        
        # Update center for geometric calculations
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Extract depth value
        center_roi = depth_map[y1:y2, x1:x2]
        if center_roi.size == 0: return 5.0, "Ahead", (0,0,0,0)
        
        # Use 90th percentile to avoid outlier noise
        depth_val = np.percentile(center_roi, 90)
        if depth_val <= 0: return 5.0, "Ahead", (0,0,0,0)

        # ====================================================
        # STEP 1: CALCULATE Z (POLYNOMIAL REGRESSION)
        # ====================================================
        # Calibrated coefficients
        A, B, C = -0.13094, 0.33910, 1.56904
        Z_meters = (A * (depth_val ** 2)) + (B * depth_val) + C
        
        # Clamp values for stability
        if Z_meters < 0.3: Z_meters = 0.3
        if Z_meters > 5.0: Z_meters = 5.0

        # ====================================================
        # STEP 2: CALCULATE RHO (SPHERICAL GEOMETRY)
        # ====================================================
        # Using dynamic center_x, center_y for object localization
        
        FOV_DEG = 60.0
        f_pixel = (w / 2) / np.tan(np.deg2rad(FOV_DEG / 2))
        
        x_norm = (center_x - (w / 2)) / f_pixel
        y_norm = (center_y - (h / 2)) / f_pixel
        
        rho_meters = Z_meters * np.sqrt(1 + x_norm**2 + y_norm**2)

        # ====================================================
        # DETERMINE DIRECTION (LEFT / RIGHT / AHEAD)
        # ====================================================
        # Vietnamese labels used for final user feedback (maintained via lang logic)
        # Internal logic/labels use English labels as per policy
        position_label = "Ahead"
        if center_x < w * 0.33:
            position_label = "Left"
        elif center_x > w * 0.66:
            position_label = "Right"

        # Localization mapping for user feedback (Vietnamese strings are handled in app.py logic)
        # We return the label which the API will translate for speech output
        direction_map = {
            "Ahead": "Phía trước",
            "Left": "Bên Trái",
            "Right": "Bên Phải"
        }
        position_text = direction_map.get(position_label, "Phía trước")

        # Smoothing
        self.history.append(rho_meters)
        smooth_dist = sum(self.history) / len(self.history)
        
        return round(smooth_dist, 2), position_text, (x1, y1, x2, y2)
