
import requests
import os, sys
import subprocess
from colorama import Fore, Back, Style
from tqdm import tqdm
from pip._internal import main as pip_main
from folder_paths import base_path
from pathlib import Path

try:
    print(Fore.GREEN + 'FaceDetailer: ' + f'{Fore.WHITE}Installing requirements' + Fore.RESET)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", f"{base_path}\\custom_nodes\\dz_facedetailer\\requirements.txt"])
except:
    print(Fore.RED + 'FaceDetailer: ' + f'{Fore.WHITE}Installing requirements failed' + Fore.RESET)

model = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"

save_loc = f"{base_path}\models\dz_facedetailer\yolo\/face_yolov8n.pt"

save_dir = os.path.dirname(save_loc)

def download_model():
    if Path(save_loc).is_file():
        print(Fore.GREEN + 'FaceDetailer: ' + f'{Fore.WHITE}Model already exists' + Fore.RESET)
    else:
        print(Fore.RED + 'FaceDetailer: ' + f'{Fore.WHITE}Model doesnt exist' + Fore.RESET)
        print(Fore.GREEN + 'FaceDetailer: ' + f'{Fore.WHITE}Downloading model' + Fore.RESET)
        response = requests.get(model, stream=True)

        try:
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kibibyte

                # tqdm will display a progress bar
                with open(save_loc, 'wb') as file, tqdm(
                    desc='Downloading',
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(block_size):
                        bar.update(len(data))
                        file.write(data)

                print(Fore.GREEN + 'FaceDetailer: ' + f'{Fore.WHITE}Model dowload finished' + Fore.RESET)
        except requests.exceptions.RequestException as err:
            print(Fore.RED + 'FaceDetailer: ' + f'{Fore.WHITE}Model download failed: {err}' + Fore.RESET)
            print(Fore.RED + 'FaceDetailer: ' + f'{Fore.WHITE}Download it manually from: {model}' + Fore.RESET)
            print(Fore.RED + 'FaceDetailer: ' + f'{Fore.WHITE}And put it in /comfyui/models/dz_facedetailer/yolo/' + Fore.RESET)
        except Exception as e:
            print(Fore.RED + 'FaceDetailer: ' + f'{Fore.WHITE}An unexpected error occurred: {e}' + Fore.RESET)

if not os.path.exists(save_dir):
    print(Fore.GREEN + 'FaceDetailer: ' + f'{Fore.WHITE}Creating models dir' + Fore.RESET)
    os.makedirs(save_dir)
else:
    print(Fore.GREEN + 'FaceDetailer: ' + f'{Fore.WHITE}Model dir already exists' + Fore.RESET)
    download_model()

from .FaceDetailer import FaceDetailer

NODE_CLASS_MAPPINGS = {
    "DZ_Face_Detailer": FaceDetailer,
}


