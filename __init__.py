import requests
import os, sys
import subprocess
from tqdm import tqdm
from pip._internal import main as pip_main
from pathlib import Path
from folder_paths import models_dir

try:
    import mediapipe
except:
    print('FaceDetailer: Installing requirements')
    my_path = os.path.dirname(__file__)
    subprocess.check_call([sys.executable, "-s", "-m", "pip", "install", "-r", os.path.join(my_path, "requirements.txt")])

model_url = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"

save_loc = os.path.join(models_dir, "dz_facedetailer", "yolo")

def download_model():
    if Path(os.path.join(save_loc, "face_yolov8n.pt")).is_file():
        print('FaceDetailer: Model already exists')
    else:
        print('FaceDetailer: Model doesnt exist')
        print('FaceDetailer: Downloading model')
        response = requests.get(model_url, stream=True)

        try:
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte

                # tqdm will display a progress bar
                with open(os.path.join(save_loc, "face_yolov8n.pt"), 'wb') as file, tqdm(
                    desc='Downloading',
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(block_size):
                        bar.update(len(data))
                        file.write(data)

                print('FaceDetailer: Model download finished')
        except requests.exceptions.RequestException as err:
            print('FaceDetailer: Model download failed: {err}')
            print(f'FaceDetailer: Download it manually from: {model_url}')
            print('FaceDetailer: And put it in /comfyui/models/dz_facedetailer/yolo/')
        except Exception as e:
            print(f'FaceDetailer: An unexpected error occurred: {e}')

if not os.path.exists(save_loc):
    print('FaceDetailer: Creating models directory')
    os.makedirs(save_loc, exist_ok=True)
    download_model()
else:
    print('FaceDetailer: Model directory already exists')
    download_model()

from .DZFaceDetailer import FaceDetailer

NODE_CLASS_MAPPINGS = {
    "DZ_Face_Detailer": FaceDetailer,
}


