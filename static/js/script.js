const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const textVi = document.getElementById('text-vi');
const textDist = document.getElementById('text-dist');
const distLine = document.getElementById('dist-line');
const status = document.getElementById('status');
const unitSelect = document.getElementById('distanceUnit');
const langSelect = document.getElementById('langSelect');
const langLabel = document.getElementById('lang-label');

let isProcessing = false;

// --- CẤU HÌNH ---
let lastSpokenCaption = ""; 
let lastSpeakTime = 0;      
const SPEAK_COOLDOWN = 8000; 
const WARNING_COOLDOWN = 6000; // Cảnh báo nguy hiểm: không lặp quá dày
let lastWarningText = "";
let lastWarningTime = 0;
let isSpeakingWarning = false;
const API_URL = '/predict';
const DIST_INTERVAL_MS = 5000;   // Distance-only mỗi 5s
const FULL_EVERY_N_TICKS = 2;    // Mỗi 2 tick (10s) chạy full (caption + distance)
let tickCount = 0;

// 1. Mở Webcam
navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
    .then(stream => { 
        video.srcObject = stream; 
        status.innerText = "Camera sẵn sàng.";
        video.onloadedmetadata = () => {
            // Chạy vòng lặp: 5s/lần, cứ 2 lần thì 1 lần full
            setInterval(captureAndSend, DIST_INTERVAL_MS);
        }
    })
    .catch(err => { 
        console.error(err); 
        status.innerText = "Lỗi Camera (Hãy dùng HTTPS hoặc localhost)!";
    });

function getSelectedLang() {
    const lang = (langSelect && langSelect.value) ? langSelect.value : "vi";
    if (langLabel) langLabel.innerText = lang.toUpperCase();
    return lang;
}

// 2. Hàm đọc TTS 
function smartSpeak(text, captionOnly, isWarning) {
    const now = Date.now();
    
    if (isWarning) {
        // Nếu đang đọc chính cảnh báo nguy hiểm rồi thì không restart (tránh lặp "Nguy hiểm" liên tục)
        if ('speechSynthesis' in window && (isSpeakingWarning || window.speechSynthesis.speaking)) {
            // Nếu đang nói mà không phải warning, ta vẫn cho phép ngắt lời 1 lần để cảnh báo
            if (!isSpeakingWarning) {
                forceSpeak(text, { interrupt: true, isWarning: true });
                lastWarningText = text;
                lastWarningTime = now;
            }
            return;
        }

        // Cooldown để tránh nói lại cùng một cảnh báo quá dày
        if (text === lastWarningText && (now - lastWarningTime) < WARNING_COOLDOWN) {
            return;
        }

        forceSpeak(text, { interrupt: true, isWarning: true });
        lastWarningText = text;
        lastWarningTime = now;
        lastSpeakTime = now; // reset cooldown chung
        return;
    }

    // Nếu đang đọc dở một câu, đừng chen vào (tránh cắt caption)
    if ('speechSynthesis' in window && window.speechSynthesis.speaking) {
        return;
    }

    if (captionOnly !== lastSpokenCaption) {
        forceSpeak(text, { interrupt: false });
        lastSpokenCaption = captionOnly;
        lastSpeakTime = now;
        return;
    }
    if (now - lastSpeakTime > SPEAK_COOLDOWN) {
        forceSpeak(text, { interrupt: false });
        lastSpeakTime = now;
    }
}

function forceSpeak(text, opts = { interrupt: false, isWarning: false }) {
    if ('speechSynthesis' in window) {
        // Chỉ cancel khi cần cảnh báo khẩn cấp và không phải đang nói warning
        if (opts && opts.interrupt && !isSpeakingWarning) {
            window.speechSynthesis.cancel();
        }
        const u = new SpeechSynthesisUtterance(text);
        const lang = getSelectedLang();
        u.lang = (lang === "en") ? "en-US" : "vi-VN";
        u.rate = 1.0; 
        if (opts && opts.isWarning) {
            isSpeakingWarning = true;
            u.onend = () => { isSpeakingWarning = false; };
            u.onerror = () => { isSpeakingWarning = false; };
        }
        window.speechSynthesis.speak(u);
    }
}

// 3. Hàm chụp và gửi ảnh
async function captureAndSend() {
    if (isProcessing || video.readyState !== 4) return;
    isProcessing = true;

    const selectedUnit = unitSelect.value; 
    const selectedLang = getSelectedLang();
    tickCount += 1;
    const mode = (tickCount % FULL_EVERY_N_TICKS === 0) ? "full" : "distance_only";

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
                unit: selectedUnit,
                lang: selectedLang,
                mode: mode
            })
        });
        const result = await response.json();
        
        // Chỉ cập nhật caption khi chạy chế độ FULL và có caption thật.
        // Với mode=distance_only, giữ nguyên caption cũ để tránh "nhấp nháy" / mất nội dung khi TTS đang đọc.
        if (mode === "full" && result.caption_vi) {
            textVi.innerText = result.caption_vi;
        }
        
        let isDanger = false;
        
        if (result.warning) {
            textDist.innerText = result.warning + " (" + result.distance + ")";
            distLine.className = "danger"; 
            document.body.style.backgroundColor = "#ffecec"; 
            isDanger = true;
    
            // ---- BỔ SUNG 3 DÒNG NÀY ----
            // Nếu đang quét nhanh mà thấy vật cản, ép chu kỳ tiếp theo phải là FULL
            if (mode === "distance_only") {
                tickCount = FULL_EVERY_N_TICKS - 1; 
            }
            // ----------------------------
    
        } else {
            textDist.innerText = (selectedLang === "en")
                ? ("Distance: " + result.distance)
                : ("Khoảng cách: " + result.distance);
            distLine.className = "normal"; 
            document.body.style.backgroundColor = "white";
        }

        // Nếu distance_only thì không có caption mới -> dùng caption gần nhất để kiểm soát cooldown
        const captionOnly = result.caption_vi || lastSpokenCaption;
        smartSpeak(result.final_speech, captionOnly, isDanger);

    } catch (error) {
        // Bắt lỗi khi mất kết nối Server
        if (error.message.includes('Failed to fetch')) {
            console.warn("Mất kết nối tới Server. Đang thử lại...");
            document.getElementById('status').innerText = "Mất kết nối Server. Đang chờ...";
        } else {
            console.error("Lỗi hệ thống:", error);
        }
    } finally {
        isProcessing = false;
    }
}