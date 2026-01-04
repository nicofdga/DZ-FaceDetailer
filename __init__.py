import logging
import os
from pathlib import Path

import requests
from folder_paths import models_dir
from tqdm import tqdm

__version__ = "2.0.0"

# Configure logging
logger = logging.getLogger("DZ-FaceDetailer")
logger.setLevel(logging.INFO)

# Model configuration
model_url = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"
save_loc = os.path.join(models_dir, "dz_facedetailer", "yolo")


def download_model():
    """Download the YOLOv8 face detection model if not present."""
    model_path = Path(os.path.join(save_loc, "face_yolov8n.pt"))

    if model_path.is_file():
        logger.info("YOLOv8 face model already exists at %s", model_path)
        return

    logger.info("YOLOv8 face model not found, downloading from HuggingFace...")

    try:
        response = requests.get(model_url, stream=True)

        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte

            # tqdm will display a progress bar
            with open(model_path, 'wb') as file, tqdm(
                desc='Downloading YOLOv8 face model',
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(block_size):
                    bar.update(len(data))
                    file.write(data)

            logger.info("YOLOv8 face model download completed successfully")
        else:
            logger.error("Failed to download model: HTTP %d", response.status_code)
            logger.error("Download it manually from: %s", model_url)
            logger.error("And place it in: %s", save_loc)

    except requests.exceptions.RequestException as err:
        logger.error("Model download failed: %s", err)
        logger.error("Download it manually from: %s", model_url)
        logger.error("And place it in: %s", save_loc)
    except Exception as e:
        logger.error("An unexpected error occurred during download: %s", e)


# Ensure model directory exists and download model if needed
if not os.path.exists(save_loc):
    logger.info("Creating models directory at %s", save_loc)
    os.makedirs(save_loc, exist_ok=True)
    download_model()
else:
    download_model()

# Import and register the ComfyUI node
from .DZFaceDetailer import FaceDetailer

NODE_CLASS_MAPPINGS = {
    "DZ_Face_Detailer": FaceDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DZ_Face_Detailer": "DZ Face Detailer",
}
