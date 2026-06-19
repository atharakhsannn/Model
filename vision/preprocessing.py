import cv2
import numpy as np
import torch
from typing import Tuple

def resize_image(image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resizes the image to the specified size."""
    if image is None:
        raise ValueError("Provided image is None.")
    return cv2.resize(image, size)

def normalize_image(image: np.ndarray, use_standard_norm: bool = False) -> np.ndarray:
    """
    Normalizes the image.
    If use_standard_norm is True, applies ImageNet mean and std normalization.
    Otherwise, simply scales to [0, 1].
    """
    if image is None:
        raise ValueError("Provided image is None.")
        
    img_float = image.astype(np.float32) / 255.0
    
    if use_standard_norm:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_float = (img_float - mean) / std
        
    return img_float

def prepare_tensor(image: np.ndarray) -> torch.Tensor:
    """Converts a numpy image (H, W, C) to a PyTorch float32 tensor (C, H, W)."""
    if image is None:
        raise ValueError("Provided image is None.")
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    return tensor
