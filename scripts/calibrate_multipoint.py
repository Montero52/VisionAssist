import cv2
import torch
import numpy as np
import os
import sys
import time

# Import class đã tối ưu
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from src.main.distance import DepthEstimator
except ImportError:
    print("Lỗi: Không tìm thấy src/main/distance.py")
    exit(1)

def analyze_system():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CHUYÊN GIA PHÂN TÍCH ĐANG CHẠY TRÊN: {device.upper()}")
    print("Đã kích hoạt chế độ TỐI ƯU HÓA FPS (Frame Skipping)")
    
    estimator = DepthEstimator(device=device)
    cap = cv2.VideoCapture(0)
    
    # Giảm độ phân giải đầu vào xuống 320x240 để tăng tốc độ xử lý cho máy yếu
    # Nếu muốn nét hơn thì để 640x480, nhưng 320x240 là mượt nhất
    cap.set(3, 320)
    cap.set(4, 240)

    print("\n=== BẮT ĐẦU QUY TRÌNH KIỂM THỬ ===")
    print("1. Di chuyển vật từ TÂM ra GÓC để xem sự chênh lệch Z vs Rho.")
    print("2. Đưa tay vào nhanh để test độ nhạy Auto-Scan.")
    print("==================================\n")

    # --- CẤU HÌNH TỐI ƯU ---
    SKIP_FRAMES = 4  # Số khung hình bỏ qua (Máy càng yếu thì tăng số này lên, vd: 5 hoặc 6)
    frame_count = 0
    
    # Biến lưu trữ kết quả tạm thời (để hiển thị khi bỏ qua frame)
    last_rho = 0.0
    last_pos = "Dang khoi tao..."
    last_box = (0, 0, 0, 0)
    ai_fps = 0.0 # FPS thực tế của việc xử lý AI

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Tăng kích thước khung hình hiển thị lên gấp đôi cho dễ nhìn (nếu input là 320x240)
        display_frame = cv2.resize(frame, (640, 480)) 
        
        # Tỷ lệ scale để vẽ box đúng vị trí (vì ta vẽ lên ảnh 640x480 nhưng AI có thể chạy trên ảnh nhỏ)
        scale_x = 640 / frame.shape[1]
        scale_y = 480 / frame.shape[0]

        # --- LOGIC FRAME SKIPPING ---
        # Chỉ chạy AI khi chia hết cho (SKIP_FRAMES + 1)
        if frame_count % (SKIP_FRAMES + 1) == 0:
            start_time = time.time()
            
            # Gọi hàm ước lượng
            rho, pos, box = estimator.estimate_distance(frame)
            
            # Cập nhật giá trị mới nhất
            last_rho = rho
            last_pos = pos
            last_box = box
            
            # Tính FPS xử lý của AI
            process_time = time.time() - start_time
            if process_time > 0:
                ai_fps = 1.0 / process_time
        
        frame_count += 1

        # --- VISUALIZATION (Luôn vẽ lên mọi khung hình bằng dữ liệu last_...) ---
        
        # Giải nén box và scale lên kích thước hiển thị
        bx1, by1, bx2, by2 = last_box
        
        # Scale tọa độ box nếu ta resize ảnh hiển thị
        d_x1 = int(bx1 * scale_x)
        d_y1 = int(by1 * scale_y)
        d_x2 = int(bx2 * scale_x)
        d_y2 = int(by2 * scale_y)

        # 1. Vẽ Box
        color = (0, 255, 0) # Xanh lá (An toàn)
        if last_rho < 0.8: color = (0, 0, 255) # Đỏ (Nguy hiểm)
        
        cv2.rectangle(display_frame, (d_x1, d_y1), (d_x2, d_y2), color, 2)
        # Vẽ tâm vật thể
        cv2.circle(display_frame, ((d_x1+d_x2)//2, (d_y1+d_y2)//2), 5, (0, 255, 255), -1)

        # 2. HIỂN THỊ DỮ LIỆU PHÂN TÍCH
        # Panel thông tin nền đen
        cv2.rectangle(display_frame, (0, 0), (640, 110), (0, 0, 0), -1)
        
        # Dòng 0: FPS
        cv2.putText(display_frame, f"AI FPS: {ai_fps:.1f} (Skipping: {SKIP_FRAMES})", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Dòng 1: Kết quả cuối cùng (Rho)
        cv2.putText(display_frame, f"DIST: {last_rho:.2f} m", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Dòng 2: Vị trí & Cảnh báo
        warning = "AN TOAN" if last_rho >= 0.8 else "NGUY HIEM!"
        cv2.putText(display_frame, f"{last_pos} | {warning}", (10, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 3. PHÂN TÍCH GÓC LỆCH (GEOMETRY CHECK)
        # Tính tâm hiển thị
        disp_center_x = (d_x1 + d_x2) // 2
        offset_x = abs(disp_center_x - 320) # 320 là giữa ảnh 640
        
        cv2.putText(display_frame, f"Offset: {offset_x}px", (450, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if offset_x > 150:
            cv2.putText(display_frame, "[TEST GOC]", (450, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        cv2.imshow("SYSTEM ANALYSIS TOOL (OPTIMIZED)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    analyze_system()