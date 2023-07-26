
try:
    import ultralytics as ul
    import mediapipe as mp
except:
    print("Installing requirements...")
    my_path = os.path.dirname(__file__)

    requirements_path = os.path.join(my_path, "requirements.txt")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
    from .FaceDetailer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    import requests
    import os, sys
    import subprocess


model = "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt"

save_loc = "./custom_nodes/facedetailer/yolo/face_yolov8n.pt"

save_dir = os.path.dirname(save_loc)

if not os.path.exists(save_dir):
    print("creating face_detailer models dir because it doesn't exist")
    os.makedirs(save_dir)

print("face_yolov8n model downloading...")
response = requests.get(model)

if response.status_code == 200:
    with open(save_loc, 'wb') as file:
        file.write(response.content)
    print("Model downloading finished.")
else:
    print("Error while download! Report the issue to the repo or download the face_yolov8n.pt model manually from Bingsu hugginface's and put in models/facedetailer/")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']