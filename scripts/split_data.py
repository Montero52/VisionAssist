import json
import random
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Split-Data")

# --- 1. IMPORT FROM CONFIG ---
try:
    # Import caption_dir from config.py
    # This helps identify the folder containing data (Flickr30k or 8k)
    from config import caption_dir
except ImportError:
    logger.error("Error: config.py not found. Please place this file in the root directory.")
    sys.exit(1)

# --- 2. AUTOMATIC PATH SETUP ---
# Handle cases where caption_dir is a directory or points to a file
if os.path.isdir(caption_dir):
    WORK_DIR = caption_dir
else:
    WORK_DIR = os.path.dirname(caption_dir)

# Input file: Total JSON file created from convert_captions.py
# (Default name is captions.json)
INPUT_FILE = os.path.join(WORK_DIR, "captions.json")

# Output directory: Save in the same location
OUTPUT_DIR = WORK_DIR

def split_dataset(input_path, output_dir, train_ratio=0.8, val_ratio=0.1):
    """
    Splits a large JSON file into 3 sub-files: Train, Val, Test.
    """
    
    # 1. Check input file
    if not os.path.exists(input_path):
        logger.error(f"ERROR: Source file not found at: {input_path}")
        logger.error("   -> Did you run 'convert_captions.py'?")
        return

    logger.info(f"Reading data from: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"ERROR: Could not read JSON file. Details: {e}")
        return
    
    # Get image list (Flexible handling)
    if isinstance(data, dict) and 'images' in data:
        images = data['images']
    elif isinstance(data, list):
        images = data
    else:
        logger.error("ERROR: Invalid JSON structure (image list not found).")
        return

    # 2. Shuffle data
    logger.info("Shuffling data...")
    random.seed(42) 
    random.shuffle(images)
    
    # 3. Calculate split sizes
    total = len(images)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    # n_test is the remainder
    
    train_data = images[:n_train]
    val_data = images[n_train : n_train + n_val]
    test_data = images[n_train + n_val:]
    
    logger.info(f"Total images found: {total}")
    logger.info(f"   - Train (80%): {len(train_data)} images")
    logger.info(f"   - Val   (10%): {len(val_data)} images")
    logger.info(f"   - Test  (10%): {len(test_data)} images")
    
    # 4. Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Helper to save JSON file
    def save_json(data_list, filename):
        file_path = os.path.join(output_dir, filename)
        
        # If original data has nested structure, preserve it
        if isinstance(data, dict) and 'images' in data:
            final_content = {'images': data_list}
            for k, v in data.items():
                if k != 'images':
                    final_content[k] = v
        else:
            final_content = data_list # Flat list
            
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(final_content, f, ensure_ascii=False, indent=2)
        logger.info(f"   -> Saved: {filename}")

    # Perform save
    logger.info("Saving split files...")
    save_json(train_data, 'train_captions.json')
    save_json(val_data, 'val_captions.json')
    save_json(test_data, 'test_captions.json')
    
    logger.info(f"COMPLETED! Files are ready at: {output_dir}")
    logger.info("   You can now run 'python train.py'!")

# --- MAIN ---
if __name__ == "__main__":
    logger.info(f"Config pointing to: {WORK_DIR}")
    
    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        logger.warning(f"File '{INPUT_FILE}' not found. Please ensure captions are converted.")
            
    split_dataset(INPUT_FILE, OUTPUT_DIR)
