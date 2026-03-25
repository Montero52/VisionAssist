# VisionAssist | Real-time Multimodal AI Assistant

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![OpenVINO](https://img.shields.io/badge/Inference%20Engine-OpenVINO-0073b5.svg)](https://docs.openvino.ai/2024/en/latest/index.html)
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
*Environment: Evaluated on **unseen VizWiz test split** (real-world noisy/blurry images) using **OpenVINO optimized models** on standard Intel CPU.*

| Metric | Score / Value | Description |
| :--- | :--- | :--- |
| **CIDEr** | **28.91** | Human-like description alignment |
| **BLEU-1** | **26.59** | High object keyword relevance |
| **Inference Latency** | **~1.5s - 2.5s** | End-to-end pipeline (Full system) |
| **Memory Footprint** | **~6.8 GB** | Stable RAM usage under load |

> **Technical Insight:** The system leverages **Transfer Learning** (Pre-trained on Flickr30k, Fine-tuned on VizWiz) to specifically handle the challenges of images captured by visually impaired users, such as motion blur and poor framing. It uses **OpenVINO Runtime** with **INT8 quantization** to achieve high throughput while maintaining stability on 8GB RAM systems.

## Key Technical Highlights
### 1. Dual-Branch AI Architecture (OpenVINO Optimized)
* **Semantic Captioning (Branch A):** Uses a **ViT Encoder** and a custom **T5-style Decoder**, both converted to **OpenVINO IR** format. Inference utilizes Length-Normalized Beam Search executed via the OpenVINO Runtime engine.
* **Distance Estimation (Branch B):** Integrates **Depth Anything V2 (Small)** model, converted and optimized with OpenVINO IR. It outputs physical distance (meters) after post-processing, enabling real-time obstacle warnings.

### 2. Production-Grade MLOps Pipeline
* **Separation of Concerns:** Core inference libraries (`src/main/`) are strictly isolated from execution entry points (`scripts/`).
* **Runtime Stability:** Integrated `threading.Lock` for concurrent request handling and periodic memory cleanup (`gc.collect()`) to prevent leaks under load.
* **Real-time TTS Feedback:** Automated browser-based Web Speech API integration that converts translated Vietnamese captions into natural audio instructions.

## Tech Stack
* **Core AI:** PyTorch, Hugging Face Transformers, timm (Vision Models).
* **Inference Engine:** **OpenVINO Toolkit (CPU/iGPU)**.
* **Backend:** Python, Flask.
* **Processing:** OpenCV, NumPy, Deep Translator.
* **Frontend:** HTML5, CSS3, JavaScript (Web Speech API).

## Project Structure
```text
VisionAssist/
├── checkpoints/          # OpenVINO IR Models (.xml, .bin) - Managed via download_weights.py
│   └── ir/               # Intermediate Representation directory
├── scripts/              # Execution scripts (Train, Data Split, Convert IR)
├── src/                  # Core AI Library
│   ├── data/             # Dataset loaders & data utilities
│   └── main/             # Core inference logic (ViT, Decoder, Distance, OpenVINO Engine)
├── static/               # Frontend UI assets (CSS, JS)
├── templates/            # Web UI templates (HTML)
├── app.py                # Main Flask server entry point
├── config.py             # Global system hyper-parameters & OpenVINO paths
├── download_weights.py   # Automated script to fetch OpenVINO model IRs
├── requirements.txt      # Python dependencies
└── .gitignore            # Excludes large weights and IDE files
```

## Quick Start
### 1. Prerequisites & Installation

```bash
# Setup Virtual Environment (Example for Windows CMD)
python -m venv .venv
.venv\Scripts\activate

# Install Dependencies (Includes OpenVINO, NNCF, Optimum-Intel)
pip install -r requirements.txt
```

### 2. Model Setup (OpenVINO IR Download)
Model weights are not tracked by Git. Run the following script to download and extract the pre-quantized OpenVINO IR models into the `checkpoints/` directory structure automatically:

```bash
python download_weights.py
```

> **Important:** The system will now use the compiled OpenVINO models (`.xml` files) defined in [`config.py`](config.py), falling back to FP16 or INT8 as configured.

### 3. Configuration & Launch

```bash
# Launch the Flask Server
python app.py
```
*Once the server is running, navigate to `http://127.0.0.1:5000` in your web browser. Allow camera permissions to begin real-time analysis.*

## Training (Flickr Captioning)
### 1. Prepare captions (Flickr8k/30k)
```bash
# Convert captions.txt -> captions.json
python scripts/convert_captions.py

# Split captions.json -> train/val/test JSON files
python scripts/split_data.py
```

### 2. Train
```bash
python scripts/train.py
```

> **Dataset note:** Datasets are intentionally ignored by Git (see [`.gitignore`](.gitignore)). Place your dataset under the path expected by [`config.py`](config.py).

## Deployment Note

By default, `app.py` runs Flask in **debug mode** for local development. For production-like serving, use a WSGI server (e.g., Waitress/Gunicorn) and disable debug.

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for the full license text.

## Authors & Acknowledgments

**Trần Nhật Quý** *Lead Developer & Maintainer* | [LinkedIn](https://www.linkedin.com/in/trannhatquy) | [GitHub](https://github.com/Montero52) | [trannhatquy0@gmail.com](mailto:trannhatquy0@gmail.com)

* **Personal Extensions (v2.0+):** Independently refactored the entire project structure for MLOps standards, integrated **Depth Anything V2** for precise spatial logic, optimized the **ViT-Transformer pipeline** for real-time web inference using **OpenVINO**, and tightened the deployment rules.

**Original Capstone Team (v1.0):**
* VisionAssist originated as a Graduation Thesis at Duy Tan University. Special thanks to the initial development team for building the core data preparation and base architecture: *Hồ Hữu Quang Sang, Ngô Anh Thư, Trần Bảo Duy, Phạm Văn Nhật Trường*.

---
> **Note:** This project was developed for educational and research purposes as part of the Graduation Thesis at Duy Tan University.
