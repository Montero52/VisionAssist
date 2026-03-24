import requests
import base64
import cv2
import numpy as np
import time

def run_automated_client(total_requests=100, delay_seconds=5):
    print(f"--- Bắt đầu bắn {total_requests} Request tự động ---")
    url = "http://127.0.0.1:5000/predict"

    # Tạo sẵn một ảnh nhiễu ngẫu nhiên và mã hóa Base64
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', dummy_img)
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    for i in range(1, total_requests + 1):
        # Chuyển đổi qua lại giữa 2 chế độ
        current_mode = "full" if i % 2 == 0 else "distance_only"
        
        payload = {
            "image": img_b64,
            "mode": current_mode,
            "lang": "vi"
        }
        
        try:
            t0 = time.time()
            res = requests.post(url, json=payload)
            t1 = time.time()
            
            if res.status_code == 200:
                print(f"[{i}/{total_requests}] {current_mode.upper()} - OK ({round((t1-t0)*1000)}ms)")
            else:
                print(f"[{i}/{total_requests}] {current_mode.upper()} - LỖI: {res.status_code}")
        except Exception as e:
            print(f"Lỗi kết nối: {e}")
            
        time.sleep(delay_seconds) # Nghỉ một nhịp để mô phỏng người dùng thật

if __name__ == "__main__":
    run_automated_client(total_requests=120, delay_seconds=5) # 120 x 5s = 10 phút