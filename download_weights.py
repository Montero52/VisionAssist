import os
import gdown
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 1. Configure Google Drive file ID
FILE_ID = '1Dv-X56iR1E3DLqKSZXn5didC-0gpKNZZ'

# 2. Configure save paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_FILE = os.path.join(CHECKPOINT_DIR, "model_epoch_50.pth")

# 3. Download function
def download_model() -> None:
    # Create checkpoints directory if it doesn't exist
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        logging.info(f"Created directory: {CHECKPOINT_DIR}")

    # Check if the file already exists
    if os.path.exists(OUTPUT_FILE):
        logging.info(f"Model file already exists at: {OUTPUT_FILE}")
        return

    logging.info("Downloading model from Google Drive... (Please wait)")
    
    # gdown download URL
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    
    try:
        gdown.download(url, OUTPUT_FILE, quiet=False)
        logging.info(f"Download successful! File saved to: {OUTPUT_FILE}")
    except Exception as e:
        logging.error(f"Error downloading file: {e}")
        logging.info("Please download manually using the link in README.md")

if __name__ == '__main__':
    download_model()