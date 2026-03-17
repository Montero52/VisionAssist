# VisionAssist | Real-time Multimodal AI Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Transformers](https://img.shields.io/badge/Library-Transformers-yellow.svg)](https://huggingface.co/)
[![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)

An end-to-end computer vision and natural language processing solution designed to assist visually impaired individuals by providing real-time environmental awareness and obstacle warnings.

<!-- <div align="center">
  <img src="assets/images/demo.gif" width="750" alt="VisionAssist Demo">
  <p><i>Dual-Branch AI: Semantic Image Captioning & Spatial Distance Estimation via Web Interface</i></p>
</div> -->

---
## Project Overview
VisionAssist transforms visual scenes into actionable audio feedback. The system mitigates the environmental awareness challenges faced by the visually impaired by implementing a **Dual-Branch Inference Pipeline**. Engineered for multimodal processing, it combines state-of-the-art Image Captioning with Monocular Depth Estimation to explicitly describe surroundings and alert users of imminent physical obstacles.

## System Capabilities & Benchmarks
*Environment: Evaluated under standard webcam constraints (320x240 resolution).*

| Metric | Target / Range | Functionality | Status |
| :--- | :--- | :--- | :--- |
| **Inference Cycle** | 10 Seconds | Web-to-Server Sync | Configurable |
| **Danger Zone** | < 0.8 Meters | High-Priority TTS Alert | Active |
| **Caution Zone** | 0.8m - 1.5m | Moderate Warning | Active |
| **Safe Zone** | > 1.5 Meters | Standard Scene Description | Active |

> **Technical Insight:** By decoupling the semantic generation (ViT-Transformer) and spatial logic (Depth Anything V2), the system dynamically filters safe environments while instantly triggering alarms for close-proximity hazards.

## Key Technical Highlights
### 1. Dual-Branch AI Architecture
* **Semantic Captioning (Branch A):** Utilizes `timm` for a pre-trained **ViT (Vision Transformer)** Encoder and a custom-built **T5-style Decoder** with Cross-Attention mechanisms. It employs Length-Normalized Beam Search for accurate and natural language generation.
* **Distance Estimation (Branch B):** Integrates **Depth Anything V2 (Small)** to generate high-quality depth maps. It applies dynamic ROI (Region of Interest) scanning and geometric quadratic regression to convert raw depth into physical distance (meters).

### 2. Production-Grade MLOps Pipeline
* **Separation of Concerns:** Core inference libraries (`src/main/`) are strictly isolated from execution entry points (`scripts/`), ensuring a clean, scalable, and highly testable codebase.
* **Real-time TTS Feedback:** Automated browser-based Web Speech API integration that converts translated Vietnamese captions into natural audio instructions.
* **Runtime Stability:** Includes basic tensor cleanup and memory hygiene patterns to maintain stability during continuous inference.

## Tech Stack
* **Core AI:** PyTorch, Hugging Face Transformers, timm (Vision Models).
* **Depth Estimation:** Depth Anything V2.
* **Backend:** Python, Flask.
* **Processing:** OpenCV, NumPy, Deep Translator.
* **Frontend:** HTML5, CSS3, JavaScript (Web Speech API).

## Project Structure
```text
vision-assist/
├── checkpoints/         # Model weights (e.g. *.pth/*.pt) - excluded from Git
├── scripts/             # Execution scripts (Train, Data Split, Convert)
├── src/                 # Core AI Library
│   ├── data/            # Dataset loaders & data utilities (Flickr8k/30k style JSON)
│   └── main/            # Core inference logic (ViT, Decoder, Distance)
├── static/              # Frontend UI assets (CSS, JS)
├── templates/           # Web UI templates (HTML)
├── app.py               # Main Flask server entry point
├── config.py            # Global system hyper-parameters
├── requirements.txt     # Python dependencies
└── download_weights.py  # Automated script to fetch model weights
```

## Quick Start

### 1. Prerequisites

* **Python 3.8+**
* **Webcam** (Required for real-time capturing)
* **CUDA-enabled GPU** (Highly recommended for PyTorch acceleration)

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/Montero52/VisionAssist.git
cd VisionAssist

# Setup Virtual Environment
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Install Dependencies
pip install -r requirements.txt
```

### 3. Model Setup (Weights Downloading)

Due to file size limitations, the heavy model checkpoints are not included in the repository. Run the automated script to fetch the required weights into the `checkpoints/` directory:

```bash
python download_weights.py
```

> **Important:** The Flask app currently loads `checkpoints/vizwiz_adapted_final.pth`. If your checkpoint has a different name, update `CHECKPOINT_PATH` inside `app.py`.

### 4. Configuration & Launch

```bash
# Launch the Flask Server
python app.py
```
*Once the server is running, navigate to `http://127.0.0.1:5000` in your web browser. Allow camera permissions to begin real-time analysis.*

## Training (Flickr Captioning)

### 1. Prepare captions (Flickr8k/30k)

```bash
# Convert captions.txt -> captions.json (inside your captions folder)
python scripts/convert_captions.py

# Split captions.json -> train/val/test JSON files
python scripts/split_data.py
```

### 2. Train

```bash
python scripts/train.py
```

> **Dataset note:** Datasets are intentionally ignored by Git (see `.gitignore`). Place your dataset under the path expected by `config.py` (or adjust `DATA_ROOT`, `image_dir`, `caption_dir` accordingly).

## Deployment Note

By default, `app.py` runs Flask in **debug mode** for local development. For production-like serving, use a WSGI server (e.g., Waitress/Gunicorn) and disable debug.

## License
This project is licensed under the **MIT License**. It is free to use for academic, research, and personal purposes. See the `LICENSE` file for the full license text.

## Authors & Acknowledgments

**Trần Nhật Quý** *Lead Developer & Maintainer* | [LinkedIn](https://www.linkedin.com/in/trannhatquy) | [GitHub](https://github.com/Montero52) | [trannhatquy0@gmail.com](mailto:trannhatquy0@gmail.com)

* **Personal Extensions (v2.0+):** Independently refactored the entire project structure for MLOps standards, integrated **Depth Anything V2** for precise spatial logic, optimized the **ViT-Transformer pipeline** for real-time web inference, and tightened the deployment rules.

**Original Capstone Team (v1.0):**
* VisionAssist originated as a Graduation Thesis at Duy Tan University. Special thanks to the initial development team for building the core data preparation and base architecture: *Hồ Hữu Quang Sang, Ngô Anh Thư, Trần Bảo Duy, Phạm Văn Nhật Trường*.

---
> **Note:** This project was developed for educational and research purposes as part of the Graduation Thesis at Duy Tan University.
