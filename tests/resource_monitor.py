import psutil
import time
import csv
from datetime import datetime

def monitor_system(duration_seconds=600, interval=1):
    """Theo dõi RAM, CPU và Xung nhịp trong 10 phút."""
    file_name = "resource_log.csv"
    headers = ["Timestamp", "RAM_Usage_MB", "CPU_Usage_Percent", "CPU_Freq_MHz"]
    
    print(f"--- Bắt đầu theo dõi trong {duration_seconds/60} phút ---")
    
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            now = datetime.now().strftime("%H:%M:%S")
            ram = psutil.virtual_memory().used / (1024 * 1024) # Đổi sang MB
            cpu_usage = psutil.cpu_percent(interval=None)
            cpu_freq = psutil.cpu_freq().current # Xung nhịp hiện tại
            
            writer.writerow([now, round(ram, 2), cpu_usage, cpu_freq])
            
            # In nhanh ra màn hình để theo dõi
            print(f"[{now}] RAM: {ram:.1f}MB | CPU: {cpu_usage}% | Freq: {cpu_freq}MHz", end='\r')
            
            time.sleep(interval)

    print(f"\n--- Đã lưu log vào {file_name} ---")

if __name__ == "__main__":
    monitor_system()