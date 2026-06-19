import cv2
import numpy as np
import os

def capture_frame_from_webcam(camera_index: int = 0) -> np.ndarray:
    """Captures a single frame from the specified webcam."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with index {camera_index}")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        raise RuntimeError("Failed to read frame from webcam.")
        
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def load_image_from_path(path: str) -> np.ndarray:
    """Loads an image from the specified file path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
        
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Failed to load image from {path}")
        
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
