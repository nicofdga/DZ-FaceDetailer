# DZ FaceDetailer

## Custom Node for ComfyUI (Stable Diffusion)

DZ FaceDetailer is a custom node for the "ComfyUI" framework inspired by !After Detailer extension from auto1111, it allows you to detect faces using Mediapipe and YOLOv8n to create masks for the detected faces. This custom node enables you to generate new faces, replace faces, and perform other face manipulation tasks using Stable Diffusion AI.

![image](https://github.com/daxthin/facedetailer/assets/78769008/22caf9e4-a29d-4e7c-b6d2-f02679b0dfff)
![image](https://github.com/daxthin/facedetailer/assets/78769008/b7bfa925-c127-427d-9ade-741ddf278648)


### Table of Contents

- [Features](#features)
- [Installation](#installation)

## Features

- Face detection using Mediapipe.
- Multiple face detection support on both models
- Face mask generation for detected faces.
- Latent/sample mapping to generated masks for face manipulation.
- Generate new faces using Stable Diffusion.
- Replace faces using LoRa or embeddings etc.

## Installation
clone the repo https://github.com/daxthin/DZ-FaceDetailer.git in custom_nodes folder
