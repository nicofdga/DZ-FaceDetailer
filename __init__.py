
import requests
import os, sys
import subprocess
from colorama import Fore, Back, Style
from tqdm import tqdm
from pip._internal import main as pip_main
from pathlib import Path
from folder_paths import models_dir

try:
    import mediapipe
    from .DZFaceDetailer import FaceDetailer
except:
    print(Fore.GREEN + 'FaceDetailer: ' + f'{Fore.WHITE}Installing requirements' + Fore.RESET)
    my_path = os.path.dirname(__file__)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", os.path.join(my_path, "requirements.txt")])

model_url = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"

save_loc = os.path.join(models_dir, "DZ-FaceDetailer\yolo")

def download_model():
    if Path(save_loc).is_file():
        print(Fore.GREEN + 'FaceDetailer: ' + f'{Fore.WHITE}Model already exists' + Fore.RESET)
    else:
        print(Fore.RED + 'FaceDetailer: ' + f'{Fore.WHITE}Model doesnt exist' + Fore.RESET)
        print(Fore.GREEN + 'FaceDetailer: ' + f'{Fore.WHITE}Downloading model' + Fore.RESET)
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

                print(Fore.GREEN + 'FaceDetailer: ' + f'{Fore.WHITE}Model dowload finished' + Fore.RESET)
        except requests.exceptions.RequestException as err:
            print(Fore.RED + 'FaceDetailer: ' + f'{Fore.WHITE}Model download failed: {err}' + Fore.RESET)
            print(Fore.RED + 'FaceDetailer: ' + f'{Fore.WHITE}Download it manually from: {model_url}' + Fore.RESET)
            print(Fore.RED + 'FaceDetailer: ' + f'{Fore.WHITE}And put it in /comfyui/models/DZ-FaceDetailer/yolo/' + Fore.RESET)
        except Exception as e:
            print(Fore.RED + 'FaceDetailer: ' + f'{Fore.WHITE}An unexpected error occurred: {e}' + Fore.RESET)

if not os.path.exists(save_loc):
    print(Fore.GREEN + 'FaceDetailer: ' + f'{Fore.WHITE}Creating models dir' + Fore.RESET)
    os.makedirs(save_loc)
    download_model()
else:
    print(Fore.GREEN + 'FaceDetailer: ' + f'{Fore.WHITE}Model dir already exists' + Fore.RESET)
    download_model()

NODE_CLASS_MAPPINGS = {
    "DZ_Face_Detailer": FaceDetailer,
}


