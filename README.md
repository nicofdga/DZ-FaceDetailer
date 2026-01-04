# DZ FaceDetailer

## Custom Node for ComfyUI (Stable Diffusion)

DZ FaceDetailer is a custom node for the "ComfyUI" framework inspired by the !After Detailer extension from Auto1111. It allows you to detect faces using YOLOv8n and OpenCV to create masks for the detected faces. This custom node enables you to generate new faces, replace faces, and perform other face manipulation tasks using Stable Diffusion AI.

![image](https://github.com/daxthin/facedetailer/assets/78769008/22caf9e4-a29d-4e7c-b6d2-f02679b0dfff)
![image](https://github.com/daxthin/facedetailer/assets/78769008/b7bfa925-c127-427d-9ade-741ddf278648)

## What's New in v2.0

✨ **Major modernization update!**

- **Removed mediapipe dependency** - No more protobuf>=5 conflicts!
- **OpenCV-based face detection** - Uses opencv-contrib-python for facial landmark detection
- **Modern Python packaging** - Full pyproject.toml support
- **Python 3.10-3.13 compatible** - Updated for modern Python versions
- **NumPy 2.x compatible** - Works with latest NumPy
- **Proper logging** - No more print statements
- **Optional dependencies** - Future-ready for flash-attn and xformers
- **Cleaner installation** - No auto-installation scripts

### Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [For ComfyUI Users](#for-comfyui-users)
  - [For Standalone Use](#for-standalone-use)
- [Usage](#usage)
- [Mask Types](#mask-types)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)

## Features

- **Face detection** using YOLOv8n (fast and accurate)
- **Two mask generation methods**:
  - `box` - Simple bounding box masks
  - `face` - Precise convex hull masks using OpenCV contour detection
- **Multiple face detection** support on both mask types
- **Batch image processing** support
- **Latent/sample mapping** to generated masks for face manipulation
- **Generate new faces** using Stable Diffusion
- **Replace faces** using LoRA or embeddings
- **Mask controls**: dilate, erode, blur

## Installation

### For ComfyUI Users

#### Method 1: Clone into custom_nodes (Recommended)

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/daxthin/DZ-FaceDetailer.git
cd DZ-FaceDetailer
pip install -r requirements.txt
```

The YOLOv8 face model will be automatically downloaded on first use.

#### Method 2: Using pyproject.toml (Modern)

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/daxthin/DZ-FaceDetailer.git
cd DZ-FaceDetailer
pip install -e .
```

### For Standalone Use

```bash
git clone https://github.com/daxthin/DZ-FaceDetailer.git
cd DZ-FaceDetailer
pip install -e .
```

#### Optional Dependencies

For future-proofing with flash attention and xformers:

```bash
# Install flash-attn (optional, for future use)
pip install -e ".[flash-attn]"

# Install xformers (optional, for future use)
pip install -e ".[xformers]"

# Install all optional dependencies
pip install -e ".[all]"

# Install development dependencies
pip install -e ".[dev]"
```

## Usage

1. Add the "DZ Face Detailer" node to your ComfyUI workflow
2. Connect your model, VAE, conditioning, and latent inputs
3. Configure the node parameters:
   - **mask_type**: Choose between "box" (bounding box) or "face" (precise contour)
   - **mask_control**: Apply "dilate", "erode", or "disabled"
   - **mask_blur**: Blur the mask edges (0-100)
   - **dilate/erode_mask_value**: Control mask expansion/contraction (0-100)
4. The node outputs:
   - Processed latent with face details
   - Mask showing detected face regions

## Mask Types

### Box Mask
- Simple square bounding box around detected faces
- Fast and reliable
- Good for most face manipulation tasks
- Uses only YOLO detection

### Face Mask
- Precise convex hull following face contours
- Uses YOLO detection + OpenCV contour detection
- Better for natural-looking face blending
- Automatically falls back to bounding box if contours can't be detected

## Requirements

### Core Dependencies (automatically installed)

- Python 3.10 or higher
- PyTorch >= 2.1.0
- torchvision >= 0.16.0
- opencv-contrib-python >= 4.9.0
- ultralytics >= 8.0.0
- pillow >= 10.0.0
- numpy >= 1.26, < 3
- requests >= 2.31.0
- pyyaml >= 6.0

### ComfyUI Dependencies (must be present)

- ComfyUI framework
- Standard ComfyUI nodes

## Troubleshooting

### "cv2.face module not found"

Make sure you have `opencv-contrib-python` installed, not just `opencv-python`:

```bash
pip install opencv-contrib-python>=4.9.0
```

### YOLO model not downloading

If the automatic download fails, you can manually download the model:

1. Download from: https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt
2. Place it in: `/path/to/ComfyUI/models/dz_facedetailer/yolo/face_yolov8n.pt`

### Protobuf conflicts (SOLVED!)

If you're upgrading from v1.x which used mediapipe:

**Good news!** v2.0 completely removes mediapipe, so there are no more protobuf>=5 conflicts. Simply update to v2.0 and reinstall dependencies.

### NumPy version issues

DZ-FaceDetailer v2.0 is compatible with both NumPy 1.26+ and NumPy 2.x. If you encounter issues:

```bash
pip install "numpy>=1.26,<3"
```

## Technical Details

- **Face Detection**: YOLOv8n-face model for fast and accurate face detection
- **Mask Generation (box mode)**: Square bounding boxes centered on detected faces
- **Mask Generation (face mode)**: Gaussian blur + Otsu's thresholding + contour detection + convex hull
- **No external landmark models required**: Everything uses built-in OpenCV algorithms

## Credits

- Inspired by [!After Detailer](https://github.com/Bing-su/adetailer) extension for Auto1111
- YOLO face model from [adetailer](https://huggingface.co/Bingsu/adetailer)
- Built for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Repository

https://github.com/daxthin/DZ-FaceDetailer
