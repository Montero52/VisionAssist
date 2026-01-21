# VisionAssist – AI Assistant for the Visually Impaired
> **Real-time AI system converting visual surroundings into Vietnamese speech to assist the visually impaired.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-Web_App-000000?style=for-the-badge&logo=flask)
![Transformer](https://img.shields.io/badge/Transformer-ViT_%2B_T5-yellow?style=for-the-badge)



## Introduction
**VisionAssist** is a Computer Vision and Deep Learning project designed to help the visually impaired perceive their environment. The system combines **Image Captioning** and **Depth Estimation** capabilities, providing feedback via **Vietnamese speech** through a mobile-optimized web interface.


### Project Status
> **Current Stage:Active Development (Beta Phase)**
> Current focus: Model fine-tuning and hyperparameter optimization to improve captioning accuracy.*
---

## Key Features

* **Smart Image Captioning:** Utilizes the powerful ViT-Base/16 architecture (trained on Flickr30k) as the Encoder and a large-scale Transformer Decoder to generate highly accurate descriptions.
* **Dual-Stage Obstacle Warning:** Integrates MiDaS to detect obstacles with two safety zones: Caution (< 1.5m) and Danger (< 0.8m) for enhanced safety.
* **Vietnamese Voice Feedback:** Automatically translates descriptions into Vietnamese and converts them to speech (TTS), allowing hands-free operation.
* **Real-time Processing:** Processing pipeline optimized on Flask to ensure minimal latency.

---

## 🚧 Development Roadmap

This project is currently in the **Fine-tuning phase**. Below is the progress report:

* [x] **Phase 1: Architecture Design** (Completed)
    * Implemented ViT (Vision Transformer) for frame feature extraction.
    * Integrated T5 for caption generation.
* [x] **Phase 2: Data Pipeline** (Completed)
    * Built preprocessing pipeline for Flickr30k datasets.
* [ ] **Phase 3: Optimization (Current Focus)**
    * Fine-tuning model weights on GPU.
    * Analyzing Loss/Accuracy metrics.
* [ ] **Phase 4: Final Demo**
    * Deploy Web Interface with live camera feed.
---

## System Architecture

The system operates based on two parallel processing pipelines:

**1. Image Captioning Pipeline (ViT + T5)**
Input Image → ViT Encoder → Feature Projection → T5 Decoder → Caption (En) → Translator → Speech (Vi)

**2. Depth Estimation Pipeline (MiDaS)**
Input Image → MiDaS Model → Depth Map → Center Crop Processing → Distance Calc → Warning Logic

---

## Directory Structure

```text
VisionAssist/
├── checkpoints/        # Contains model weights (.pth)
├── src/
│   ├── main/
│   │   ├── encode/     # ViT-Base Encoder
│   │   ├── decode/     # Transformer Decoder 
│   │   ├── model/      # Combined Architecture
│   │   ├── distance.py # Distance estimation logic & ROI processing
│   └── utils/          # Utility functions
├── templates/          # Web Interface (HTML)
├── static/             # Static resources (CSS, Images)
├── app.py              # Main Flask Server
├── config.py           # Model configuration parameters
├── download_weights.py # Automatic model download script
└── requirements.txt    # List of dependencies

```

---

## Installation & Setup

### 1. Environment Preparation

Requires Python 3.8+. A virtual environment is recommended.

```bash
git clone https://github.com/Montero52/VisionAssist.git
cd VisionAssist

python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt

```

### 2. Download Model Weights (Experimental)

> **Note:** The model is currently in the **Beta Fine-tuning phase**. The weights below are provided for testing purposes only and may produce inaccurate captions in complex scenes.

* **Current Version:** `v0.5-beta` (Trained on 50 epochs)
* **Download Link:** [[Click here to download (.pth)](https://drive.usercontent.google.com/download?id=1Dv-X56iR1E3DLqKSZXn5didC-0gpKNZZ&authuser=0)]

The project requires the `model_epoch_50.pth` weight file. You can download it automatically using:

```bash
python download_weights.py
```
*To achieve the best results, it is recommended to train the model from scratch using the provided dataset.*

### 3. Run Application

```bash
python app.py

```

Access at: `http://localhost:5000`

---

## Usage Guide

1. Ensure the computer has a Webcam (or external camera connected).
2. Open a browser and access the localhost address.
3. Grant **Camera access** when prompted.
4. The system will automatically analyze the scene and read results after every processing cycle (default 6 seconds).

**Note for Mobile Usage:**
Due to browser security policies, the Camera may not work via HTTP (IP address). To test on a mobile device, using **ngrok** to create a secure HTTPS tunnel is recommended.

---

## Technologies Used

* **Deep Learning:** PyTorch, Torchvision, Timm
* **Models:** Vision Transformer (ViT-Base/16), Transformer Decoder (Custom T5-style), MiDaS (DPT-Hybrid)
* **Dataset:** Flickr30k
* **Backend:** Flask
* **Frontend:** HTML5, JavaScript (Web Speech API)
* **Tools:** OpenCV, Numpy, Deep-translator

---


## Authors & Acknowledgments

**Lead Developer & Current Maintainer:**
* **Trần Nhật Quý** ([@Montero52](https://github.com/Montero52))
    * **Education:** Senior Student - Duy Tan University (Computer Science)
    * **Contact:** [trannhatquy0@gmail.com](mailto:trannhatquy0@gmail.com)
    * **LinkedIn:** [linkedin.com/in/trannhatquy](https://www.linkedin.com/in/trannhatquy)
    * **GitHub:** [github.com/Montero52](https://github.com/Montero52)
* **Responsibility:** Responsible for architecture optimization, fine-tuning, and system integration (Jan 2026 - Present).

**Original Contributors (Capstone Project Team):**
*This project originated as a Graduation Thesis at Duy Tan University. Special thanks to the initial development team:*
* Hồ Hữu Quang Sang 
* Ngô Anh Thư 
* Trần Bảo Duy
* Phạm Văn Nhật Trường 

---
> **Note:** This project was developed for educational and research purposes as part of the Graduation Thesis at Duy Tan University.
