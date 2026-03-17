const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const textVi = document.getElementById('text-vi');
const textDist = document.getElementById('text-dist');
const distLine = document.getElementById('dist-line');
const status = document.getElementById('status');
const unitSelect = document.getElementById('distanceUnit');

let isProcessing = false;

// --- CẤU HÌNH ---
let lastSpokenCaption = ""; 
let lastSpeakTime = 0;      
const SPEAK_COOLDOWN = 8000; 
const API_URL = '/predict';

// 1. Mở Webcam
navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
    .then(stream => { 
        video.srcObject = stream; 
        status.innerText = "Camera sẵn sàng.";
        video.onloadedmetadata = () => {
            // Chụp ảnh mỗi 10 giây
            setInterval(captureAndSend, 10000); 
        }
    })
    .catch(err => { 
        console.error(err); 
        status.innerText = "Lỗi Camera (Hãy dùng HTTPS hoặc localhost)!";
    });

// 2. Hàm đọc TTS 
function smartSpeak(text, captionOnly, isWarning) {
    const now = Date.now();
    
    if (isWarning) {
        forceSpeak(text);
        lastSpeakTime = now;
        return;
    }
    if (captionOnly !== lastSpokenCaption) {
        forceSpeak(text);
        lastSpokenCaption = captionOnly;
        lastSpeakTime = now;
        return;
    }
    if (now - lastSpeakTime > SPEAK_COOLDOWN) {
        forceSpeak(text);
        lastSpeakTime = now;
    }
}

function forceSpeak(text) {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel(); 
        const u = new SpeechSynthesisUtterance(text);
        u.lang = 'vi-VN'; 
        u.rate = 1.0; 
        window.speechSynthesis.speak(u);
    }
}

// 3. Hàm chụp và gửi ảnh
async function captureAndSend() {
    if (isProcessing || video.readyState !== 4) return;
    isProcessing = true;

    const selectedUnit = unitSelect.value; 

    // Tối ưu độ phân giải 320x240
    canvas.width = 320; 
    canvas.height = 240; 

    ctx.drawImage(video, 0, 0, 320, 240);
    const imageData = canvas.toDataURL('image/jpeg', 0.5); 

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                image: imageData,
                unit: selectedUnit 
            })
        });
        const result = await response.json();
        
        textVi.innerText = result.caption_vi;
        
        let isDanger = false;
        
        if (result.warning) {
            textDist.innerText = result.warning + " (" + result.distance + ")";
            distLine.className = "danger"; 
            document.body.style.backgroundColor = "#ffecec"; 
            isDanger = true;
        } else {
            textDist.innerText = "Khoảng cách: " + result.distance;
            distLine.className = "normal"; 
            document.body.style.backgroundColor = "white";
        }

        smartSpeak(result.final_speech, result.caption_vi, isDanger);

    } catch (error) {
        console.error("Lỗi:", error);
        status.innerText = "Lỗi kết nối Server.";
    } finally {
        isProcessing = false;
    }
}