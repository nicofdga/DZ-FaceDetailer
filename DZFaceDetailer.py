import torch
from mediapipe import solutions
import cv2
import numpy as np
from PIL import Image, ImageFilter
from ultralytics import YOLO
import os
import comfy
import nodes
from folder_paths import base_path

face_model_path = os.path.join(base_path, "models/DZ-FaceDetailer/yolo/face_yolov8n.pt")
MASK_CONTROL = ["dilate", "erode", "disabled"]
MASK_TYPE = ["box", "face"]

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
        input_tensor_img = vae.decode(latent_image["samples"])
        # convert input latent to numpy array for yolo model
        img = image2nparray(input_tensor_img, False)
        # Process the face mesh or make the face box for masking
        if mask_type == "box":
            try:
                final_mask = facebox_mask(img, mask_type)
            except:
                return (latent_image, )
        else:
            try:
                final_mask = facemesh_mask(img, mask_type)
            except:
                return (latent_image, )
        # Erode/Dilate mask
        if mask_control == "dilate":
            if dilate_mask_value > 0:
                final_mask = dilate_mask(final_mask, dilate_mask_value)
        elif mask_control == "erode":
            if erode_mask_value > 0:
                final_mask = erode_mask(final_mask, erode_mask_value)
        if mask_blur > 0:
            final_mask_image = Image.fromarray(final_mask)
            blurred_mask_image = final_mask_image.filter(
                ImageFilter.GaussianBlur(radius=mask_blur))
            final_mask = np.array(blurred_mask_image)

        final_mask = np.array(Image.fromarray(
            final_mask).getchannel('A')).astype(np.float32) / 255.0
        # Convert mask to tensor and assign the mask to the input tensor
        final_mask = 1. - torch.from_numpy(final_mask)

        latent_mask = set_mask(latent_image, final_mask)

        latent = nodes.common_ksampler(
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_mask, denoise=denoise)

        return (latent[0], final_mask,)


def facebox_mask(image, mask_type):
    # setup yolov8n face detection model
    face_model = YOLO(face_model_path)
    face_bbox = face_model(image)
    boxes = face_bbox[0].boxes
    box = boxes[0].xyxy
    x_min, y_min, x_max, y_max = box[0].tolist()

    # Calculate the center of the bounding box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Calcule the maximum width and height
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

    # Create an empty image with alpha and set the square in the face location
    mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    cv2.rectangle(mask, (new_x_min, new_y_min), (new_x_max, new_y_max), (0, 0, 0, 255), -1)
    mask[:, :, 3] = ~mask[:, :, 3]  # invert the mask

    return mask


def facemesh_mask(image, mask_type):
    mp_face_mesh = solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # List of detected face points
            points = []
            for landmark in face_landmarks.landmark:
                cx, cy = int(
                    landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                points.append([cx, cy])

            # Empty image with the same shape as input
            mask = np.zeros(
                (image.shape[0], image.shape[1], 4), dtype=np.uint8)

            # Obtain the countour of the face
            convex_hull = cv2.convexHull(np.array(points))

            # Fill the contour and store it in alpha for the mask
            cv2.fillConvexPoly(mask, convex_hull, (0, 0, 0, 255))
            mask[:, :, 3] = ~mask[:, :, 3]

            return mask


def erode_mask(mask, dilate):
    # I use erode function because the mask is inverted
    # later I will fix it
    kernel = np.ones((int(dilate), int(dilate)), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    return dilated_mask


def dilate_mask(mask, erode):
    # I use dilate function because the mask is inverted like the other function
    # later I will fix it
    kernel = np.ones((int(erode), int(erode)), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    return eroded_mask


def image2nparray(image, BGR):
    """
    convert tensor image to numpy array

    Args:
        image (Tensor): Tensor image

    Returns:
        returns: Numpy array.

    """
    narray = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

    if BGR:
        return narray
    else:
        return narray[:, :, ::-1]


def set_mask(samples, mask):
    s = samples.copy()
    print(s)
    s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    return s
