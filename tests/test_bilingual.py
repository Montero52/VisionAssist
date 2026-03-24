import sys
import os
import numpy as np
from PIL import Image
from transformers import T5Tokenizer
from torchvision import transforms
from deep_translator import GoogleTranslator

# Thêm đường dẫn gốc để import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main.model.openvino_engine import OpenVINOEngine
# BỎ IMPORT TỪ APP.PY ĐỂ TRÁNH TRÀN RAM DO KHỞI TẠO LẠI TOÀN BỘ APP

# Tự định nghĩa một hàm dịch thuật cô lập cho mục đích Test
def mock_translate_en_to_vi(text):
    if not text or text.strip() == "":
        return ""
    try:
        translator = GoogleTranslator(source='en', target='vi')
        return translator.translate(text)
    except Exception as e:
        print(f"Lỗi dịch thuật API: {e}")
        return text

def test_bilingual_consistency():
    print("--- Testing Bilingual Consistency (OpenVINO -> Translation) ---")
    
    try:
        # 1. Setup & Khởi động Engine (Cực kỳ nhẹ nhàng)
        tokenizer = T5Tokenizer.from_pretrained("t5-base", legacy=False)
        ov_engine = OpenVINOEngine()
        
        # 2. Tạo ảnh Dummy nhưng có độ nhiễu (tránh model bị 'mù' do ảnh đen)
        # Sử dụng ảnh nhiễu ngẫu nhiên thay vì ảnh đen toàn tập
        dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_img = Image.fromarray(dummy_img)
        
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.685, 0.656, 0.606], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = image_transform(pil_img).unsqueeze(0).numpy()

        # 3. Chạy OpenVINO Inference
        print("Running OpenVINO Captioning on Dummy Image...")
        
        # LƯU Ý: Đảm bảo OpenVINOEngine của bạn có hàm generate_caption
        # Nếu Engine của bạn vẫn dùng beam_search, hãy giữ nguyên như cũ, 
        # nhưng tôi khuyến khích gói gọn nó trong generate_caption.
        try:
            # Giả định Engine đã gói gọn logic
            caption_en = ov_engine.generate_caption(img_tensor, tokenizer) 
        except AttributeError:
            # Nếu Engine chưa có hàm generate_caption, dùng lại beam_search
            caption_en = ov_engine.beam_search(img_tensor, tokenizer, beam_size=1, max_len=20)
            
        print(f"Original EN: '{caption_en}'")

        # 4. Kiểm tra dịch thuật cho Dummy Image
        if caption_en:
            caption_vi = mock_translate_en_to_vi(caption_en)
            print(f"Translated VI: '{caption_vi}'")
        else:
            print("WARNING: OpenVINO produced empty caption on Dummy Image.")

        # 5. Bài test chất lượng (Đưa các câu chuẩn VizWiz vào)
        test_sentences = [
            "a person holding a white cane",
            "a car parked on the side of the road",
            "a group of people walking across a street"
        ]
        
        print("\n--- Batch Translation Quality Test ---")
        for sent in test_sentences:
            vi = mock_translate_en_to_vi(sent)
            print(f"EN: {sent} \n-> VI: {vi}\n")

        print("--- TEST COMPLETED ---")

    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    test_bilingual_consistency()