import cv2
import torch
import numpy as np
import os
import sys
import time

# Thêm đường dẫn import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.main.distance import DepthEstimator
except ImportError:
    print("❌ Lỗi Import distance.py")
    exit(1)

def run_multipoint_calibration():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Đang khởi tạo Model trên: {device.upper()}...")
    
    # Scale=1.0 để lấy Raw thuần túy
    estimator = DepthEstimator(device=device, scale_factor=1.0)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # DANH SÁCH CÁC MỐC CẦN ĐO (Bạn cần chuẩn bị thước dây)
    # Chúng ta sẽ đo 4 điểm quan trọng nhất cho người khiếm thị
    TARGET_DISTANCES = [0.5, 1.0, 1.5, 2.0] # Đơn vị: Mét
    collected_raws = []
    
    current_step = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- XỬ LÝ AI ---
        inputs = estimator.processor(images=frame, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)
        with torch.no_grad():
            depth_raw = estimator.model(pixel_values).predicted_depth
            depth_raw = torch.nn.functional.interpolate(
                depth_raw.unsqueeze(1), size=frame.shape[:2],
                mode="bicubic", align_corners=False
            ).squeeze()
        depth_map = depth_raw.cpu().numpy()

        # --- ROI (Giữ Center 0.5 để ổn định nhất) ---
        h, w = frame.shape[:2]
        roi_h, roi_w = int(h * 0.2), int(w * 0.2)
        center_y, center_x = int(h * 0.5), int(w * 0.5) 
        
        y1, y2 = center_y - roi_h//2, center_y + roi_h//2
        x1, x2 = center_x - roi_w//2, center_x + roi_w//2
        
        roi_depth = depth_map[y1:y2, x1:x2]
        current_raw = np.median(roi_depth) if roi_depth.size > 0 else 0.0

        # --- GIAO DIỆN ---
        # Depth Map màu
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-5)
        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Hướng dẫn
        cv2.putText(frame, f"RAW HIEN TAI: {current_raw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if current_step < len(TARGET_DISTANCES):
            target = TARGET_DISTANCES[current_step]
            msg1 = f"BUOC {current_step + 1}/{len(TARGET_DISTANCES)}: Dat vat cach {target} MET"
            msg2 = "Nhan phim 'c' de lay mau"
            cv2.putText(frame, msg1, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, msg2, (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "DA XONG! Dang tinh toan...", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        combined = np.hstack((frame, depth_color))
        cv2.imshow("Multi-Point Calibration", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        
        if key == ord('c') and current_step < len(TARGET_DISTANCES):
            if current_raw <= 0:
                print("⚠️ Giá trị Raw không hợp lệ!")
                continue
                
            print(f"✅ Đã lưu mẫu {TARGET_DISTANCES[current_step]}m: Raw = {current_raw:.4f}")
            collected_raws.append(current_raw)
            current_step += 1
            time.sleep(0.5)
            
            # KHI ĐÃ ĐỦ MẪU -> TÍNH TOÁN NGAY
            if current_step == len(TARGET_DISTANCES):
                print("\n" + "="*50)
                print("🧮 ĐANG CHẠY HỒI QUY ĐA THỨC (POLYNOMIAL REGRESSION)...")
                
                # Dữ liệu: X = Raw (đầu vào), Y = Distance (đầu ra mong muốn)
                X = np.array(collected_raws)
                Y = np.array(TARGET_DISTANCES)
                
                # Để khớp tốt hơn, ta thường hồi quy theo nghịch đảo (1/Raw) hoặc Raw tùy model
                # Với Depth Anything, quan hệ giữa Raw và Mét khá phức tạp.
                # Ta thử dùng hồi quy bậc 2 trực tiếp: Dist = A*Raw^2 + B*Raw + C
                # Hoặc hồi quy theo nghịch đảo: Dist = A * (1/Raw) + B
                
                # Cách 1: Hồi quy đa thức bậc 2 (Parabola) - Thường khớp tốt nhất với dữ liệu thực tế
                coeffs = np.polyfit(X, Y, 2) # Bậc 2
                A, B, C = coeffs
                
                print("="*50)
                print("🎉 KẾT QUẢ PHƯƠNG TRÌNH CHUẨN XÁC 🎉")
                print(f"Phương trình: Distance = ({A:.5f} * Raw^2) + ({B:.5f} * Raw) + {C:.5f}")
                print("-" * 50)
                print("👉 Hãy copy đoạn code dưới đây vào hàm estimate_distance trong file distance.py:")
                print("-" * 50)
                print(f"""
        # --- CODE COPY ---
        # Phương trình hồi quy đa thức bậc 2
        # A={A:.5f}, B={B:.5f}, C={C:.5f}
        
        if depth_val <= 0: return 9.9
        
        # Áp dụng công thức Parabola
        Z_meters = ({A:.5f} * (depth_val ** 2)) + ({B:.5f} * depth_val) + {C:.5f}
        
        # Giới hạn an toàn (nếu Raw lạ quá thì không báo âm)
        if Z_meters < 0.3: Z_meters = 0.3
        if Z_meters > 5.0: Z_meters = 5.0
        
        return Z_meters
        # -----------------
                """)
                print("="*50 + "\n")
                
                # Giữ màn hình để đọc
                cv2.waitKey(0)
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_multipoint_calibration()