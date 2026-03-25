import os
import gdown
import logging
import zipfile

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 1. Google Drive File ID (Paste your new ZIP file ID here)
FILE_ID = '1WqfIS8u_x4bogdKbCPcwGEDsfnzMAooW'

# 2. Directory Configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
IR_DIR = os.path.join(CHECKPOINT_DIR, "ir")
ZIP_FILE_PATH = os.path.join(CHECKPOINT_DIR, "ir_weights.zip")

def download_and_setup_model() -> None:
    """
    Downloads the OpenVINO IR weights (ZIP format) from Google Drive, 
    extracts them to preserve the directory structure, and cleans up.
    """
    # Ensure the base checkpoints directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    logging.info(f"Verified base directory at: {CHECKPOINT_DIR}")

    # Check if the 'ir' directory already exists and contains files to avoid redundant downloads
    if os.path.exists(IR_DIR) and len(os.listdir(IR_DIR)) > 0:
        logging.info("Model weights are already set up. Skipping download.")
        return

    logging.info("Downloading OpenVINO IR models from Google Drive... (This may take a moment)")
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    
    try:
        # Download the compressed weights
        gdown.download(url, ZIP_FILE_PATH, quiet=False)
        logging.info("Download completed successfully! Extracting directory structure...")
        
        # Extract the ZIP file directly into the checkpoints directory
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(CHECKPOINT_DIR)
        
        # Remove the ZIP file to free up disk space
        os.remove(ZIP_FILE_PATH)
        logging.info("Extraction complete and temporary files cleaned up. Environment is ready!")
        
    except Exception as e:
        logging.error(f"An error occurred during the setup process: {e}")
        logging.info("Please verify the FILE_ID and ensure the Google Drive link is accessible.")

if __name__ == '__main__':
    download_and_setup_model()