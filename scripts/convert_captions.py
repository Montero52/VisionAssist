import json
import os
import csv
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
logger = logging.getLogger("CaptionConverter")

# --- 1. IMPORT FROM CONFIG ---
try:
    from config import caption_dir
except ImportError:
    logger.error("Error: config.py not found.")
    sys.exit(1)

# --- 2. SETUP PATHS ---
logger.info(f"Working directory: {caption_dir}")

# Define INPUT (Source TXT file)
INPUT_TXT_FILE = os.path.join(caption_dir, "captions.txt")

# Define OUTPUT (Target JSON file)
OUTPUT_JSON_FILE = os.path.join(caption_dir, "captions.json") 

logger.info(f"Reading from: {INPUT_TXT_FILE}")
logger.info(f"Writing to: {OUTPUT_JSON_FILE}")

folder_path = os.path.dirname(OUTPUT_JSON_FILE)
# Re-confirming input path relative to folder_path
INPUT_TXT_FILE = os.path.join(folder_path, "captions.txt")

# Check if source file exists to avoid errors
if not os.path.exists(INPUT_TXT_FILE):
    logger.warning(f"WARNING: Source file not found at {INPUT_TXT_FILE}")

def convert_txt_to_json():
    logger.info(f"Target folder: {folder_path}")
    
    # Check source file again
    if not os.path.exists(INPUT_TXT_FILE):
        logger.error(f"ERROR: Source file not found at:\n   {INPUT_TXT_FILE}")
        logger.error("Please ensure you have renamed the original file to 'captions.txt' and placed it in the captions directory.")
        return

    logger.info(f"🔄 Reading source file: {os.path.basename(INPUT_TXT_FILE)}...")
    
    temp_dict = {}
    count_skipped = 0

    try:
        with open(INPUT_TXT_FILE, 'r', encoding='utf-8') as f:
            # Flickr8k/30k data usually comma-separated
            reader = csv.reader(f, delimiter=',') 
            
            # Try to skip header (if first line is image,caption)
            first_line = next(reader, None)
            if first_line and "image" not in first_line[0].lower():
                # If no header, seek back to start
                f.seek(0)
                reader = csv.reader(f, delimiter=',')

            for row in reader:
                if len(row) < 2:
                    count_skipped += 1
                    continue
                
                img_name = row[0].strip()
                # Join remaining parts as caption (in case caption contains commas)
                caption = ",".join(row[1:]).strip()
                
                # Handle old style image names (image.jpg#0)
                if "#" in img_name:
                    img_name = img_name.split("#")[0]

                if img_name not in temp_dict:
                    temp_dict[img_name] = []
                
                # Only add caption if not already present (avoid duplicates)
                if caption not in temp_dict[img_name]:
                    temp_dict[img_name].append(caption)

    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return

    # Convert to list object format
    final_data = []
    for img, caps in temp_dict.items():
        entry = {
            "file_name": img,
            "captions": caps
        }
        final_data.append(entry)

    logger.info(f"Processed {len(final_data)} images.")
    if count_skipped > 0:
        logger.info(f"Skipped {count_skipped} invalid/empty lines.")

    # Create destination directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_JSON_FILE), exist_ok=True)

    # Save JSON file
    with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    logger.info(f"SUCCESS! Standard JSON file saved at:\n   {OUTPUT_JSON_FILE}")
    logger.info("You can now run 'python train.py'!")

if __name__ == "__main__":
    convert_txt_to_json()
