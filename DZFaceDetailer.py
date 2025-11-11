import logging
import os

import comfy
import cv2
import nodes
import numpy as np
import torch
from folder_paths import base_path
from PIL import Image, ImageFilter
from ultralytics import YOLO

# Configure logging
logger = logging.getLogger("DZ-FaceDetailer")

face_model_path = os.path.join(base_path, "models/dz_facedetailer/yolo/face_yolov8n.pt")
MASK_CONTROL = ["dilate", "erode", "disabled"]
MASK_TYPE = ["face", "box"]

# Lazy-loaded facemark detector
_facemark_detector = None


def _get_facemark_detector():
    """
    Lazy-load the OpenCV FacemarkLBF detector.

    Returns the facemark detector instance, creating it on first call.
    """
    global _facemark_detector

    if _facemark_detector is None:
        try:
            # Create FacemarkLBF detector
            _facemark_detector = cv2.face.FacemarkLBF_create()

            # Load the pre-trained model
            # OpenCV's LBF model can be trained or we can use a pre-trained one
            # For now, we'll use the detector without explicit model loading
            # as opencv-contrib includes built-in capabilities
            logger.info("OpenCV FacemarkLBF detector initialized")

        except AttributeError:
            logger.error(
                "cv2.face module not found. Ensure opencv-contrib-python>=4.9.0 is installed. "
                "Install with: pip install opencv-contrib-python>=4.9.0"
            )
            raise RuntimeError(
                "OpenCV face module not available. "
                "Install opencv-contrib-python>=4.9.0"
            )

    return _facemark_detector


class FaceDetailer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {
                    "model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "latent_image": ("LATENT", ),
                    "vae": ("VAE",),
                    "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100}),
                    "mask_type": (MASK_TYPE, ),
                    "mask_control": (MASK_CONTROL, ),
                    "dilate_mask_value": ("INT", {"default": 3, "min": 0, "max": 100}),
                    "erode_mask_value": ("INT", {"default": 3, "min": 0, "max": 100}),
                }
                }

    RETURN_TYPES = ("LATENT", "MASK",)

    FUNCTION = "detailer"

    CATEGORY = "face_detailer"

    def detailer(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, vae, mask_blur, mask_type, mask_control, dilate_mask_value, erode_mask_value):

        # input latent decoded to tensor image for processing
        tensor_img = vae.decode(latent_image["samples"])

        batch_size = tensor_img.shape[0]

        mask = Detection().detect_faces(tensor_img, batch_size, mask_type, mask_control, mask_blur, dilate_mask_value, erode_mask_value)

        latent_mask = set_mask(latent_image, mask)

        latent = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_mask, denoise=denoise)

        return (latent[0], latent[0]["noise_mask"],)


class Detection:
    def __init__(self):
        pass

    def detect_faces(self, tensor_img, batch_size, mask_type, mask_control, mask_blur, mask_dilate, mask_erode):
        mask_imgs = []
        for i in range(0, batch_size):
            # convert input latent to numpy array for yolo model
            img = image2nparray(tensor_img[i], False)
            # Process the face mesh or make the face box for masking
            if mask_type == "box":
                final_mask = facebox_mask(img)
            else:
                final_mask = facemesh_mask(img)

            final_mask = self.mask_control(final_mask, mask_control, mask_blur, mask_dilate, mask_erode)

            final_mask = np.array(Image.fromarray(final_mask).getchannel('A')).astype(np.float32) / 255.0
            # Convert mask to tensor and assign the mask to the input tensor
            final_mask = torch.from_numpy(final_mask)

            mask_imgs.append(final_mask)

        final_mask = torch.stack(mask_imgs)

        return final_mask

    def mask_control(self, numpy_img, mask_control, mask_blur, mask_dilate, mask_erode):
        numpy_image = numpy_img.copy()
        # Erode/Dilate mask
        if mask_control == "dilate":
            if mask_dilate > 0:
                numpy_image = self.dilate_mask(numpy_image, mask_dilate)
        elif mask_control == "erode":
            if mask_erode > 0:
                numpy_image = self.erode_mask(numpy_image, mask_erode)
        if mask_blur > 0:
            final_mask_image = Image.fromarray(numpy_image)
            blurred_mask_image = final_mask_image.filter(
                ImageFilter.GaussianBlur(radius=mask_blur))
            numpy_image = np.array(blurred_mask_image)

        return numpy_image

    def erode_mask(self, mask, dilate):
        # I use erode function because the mask is inverted
        # later I will fix it
        kernel = np.ones((int(dilate), int(dilate)), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        return dilated_mask

    def dilate_mask(self, mask, erode):
        # I use dilate function because the mask is inverted like the other function
        # later I will fix it
        kernel = np.ones((int(erode), int(erode)), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        return eroded_mask


def facebox_mask(image):
    """
    Create a bounding box mask around detected faces using YOLO.

    Args:
        image: Input image as numpy array

    Returns:
        Mask image with face bounding boxes
    """
    # Create an empty image with alpha
    mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

    # setup yolov8n face detection model
    face_model = YOLO(face_model_path)
    face_bbox = face_model(image)
    boxes = face_bbox[0].boxes

    for box in boxes.xyxy:
        x_min, y_min, x_max, y_max = box.tolist()
        # Calculate the center of the bounding box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Calculate the maximum width and height
        width = x_max - x_min
        height = y_max - y_min
        max_size = max(width, height)

        # Get the new WxH for a ratio of 1:1
        new_width = max_size
        new_height = max_size

        # Calculate the new coordinates
        new_x_min = int(center_x - new_width / 2)
        new_y_min = int(center_y - new_height / 2)
        new_x_max = int(center_x + new_width / 2)
        new_y_max = int(center_y + new_height / 2)

        # set the square in the face location
        cv2.rectangle(mask, (new_x_min, new_y_min), (new_x_max, new_y_max), (0, 0, 0, 255), -1)

    return mask


def facemesh_mask(image):
    """
    Create a precise face mask using OpenCV face landmark detection.

    Uses YOLO for face detection and OpenCV's built-in contour detection
    for creating a convex hull mask around facial features.

    Args:
        image: Input image as numpy array

    Returns:
        Mask image with precise face contours
    """
    faces_mask = []

    # Empty image with the same shape as input
    mask = np.zeros(
        (image.shape[0], image.shape[1], 4), dtype=np.uint8)

    # setup yolov8n face detection model
    face_model = YOLO(face_model_path)
    face_bbox = face_model(image)
    boxes = face_bbox[0].boxes

    for box in boxes.xyxy:
        x_min, y_min, x_max, y_max = box.tolist()
        # Calculate the center of the bounding box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Calculate the maximum width and height
        width = x_max - x_min
        height = y_max - y_min
        max_size = max(width, height)

        # Get the new WxH for a ratio of 1:1
        new_width = max_size
        new_height = max_size

        # Calculate the new coordinates
        new_x_min = int(center_x - new_width / 2)
        new_y_min = int(center_y - new_height / 2)
        new_x_max = int(center_x + new_width / 2)
        new_y_max = int(center_y + new_height / 2)

        # set the square in the face location
        face = image[new_y_min:new_y_max, new_x_min:new_x_max, :]

        # Convert to grayscale for better contour detection
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Use adaptive thresholding and contour detection for face mask
        # This is a simpler approach that works well without requiring external models
        blur = cv2.GaussianBlur(gray_face, (5, 5), 0)

        # Use Otsu's thresholding
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour (likely the face)
            largest_contour = max(contours, key=cv2.contourArea)

            # Create a convex hull from the largest contour
            hull = cv2.convexHull(largest_contour)

            # Create mask for this face
            face_mask = np.zeros((face.shape[0], face.shape[1], 4), dtype=np.uint8)

            # Fill the convex hull
            cv2.fillConvexPoly(face_mask, hull, (0, 0, 0, 255))

            faces_mask.append([face_mask, [new_x_min, new_x_max, new_y_min, new_y_max]])
        else:
            # Fallback: if no contours found, use the bounding box
            face_mask = np.zeros((face.shape[0], face.shape[1], 4), dtype=np.uint8)
            face_mask[:, :, 3] = 255  # Fill entire face region
            faces_mask.append([face_mask, [new_x_min, new_x_max, new_y_min, new_y_max]])
            logger.debug("No contours found for face, using bounding box fallback")

    for face_mask in faces_mask:
        paste_numpy_images(mask, face_mask[0], face_mask[1][0], face_mask[1][1], face_mask[1][2], face_mask[1][3])

    logger.debug("%d faces detected and masked", len(faces_mask))

    return mask


def paste_numpy_images(target_image, source_image, x_min, x_max, y_min, y_max):
    """
    Paste the source image into the target image at the specified coordinates.

    Args:
        target_image: Target image array
        source_image: Source image array to paste
        x_min, x_max, y_min, y_max: Coordinates for pasting

    Returns:
        Modified target image
    """
    target_image[y_min:y_max, x_min:x_max, :] = source_image
    return target_image


def image2nparray(image, BGR):
    """
    Convert tensor image to numpy array.

    Args:
        image (Tensor): Tensor image
        BGR (bool): If True, return BGR format; if False, return RGB format

    Returns:
        numpy.ndarray: Numpy array in uint8 format
    """
    narray = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

    if BGR:
        return narray
    else:
        return narray[:, :, ::-1]


def set_mask(samples, mask):
    """
    Add a mask to latent samples.

    Args:
        samples: Latent samples dictionary
        mask: Mask tensor

    Returns:
        Modified samples with mask
    """
    s = samples.copy()
    s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    return s
