import os
import cv2
import torch
import numpy as np
from typing import Union, Tuple

from core.model_dme import DensityMapRegressor
from vision.preprocessing import resize_image, normalize_image, prepare_tensor

def load_model(weights_path: str, device: torch.device) -> DensityMapRegressor:
    """Loads the DensityMapRegressor model."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    model = DensityMapRegressor()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def infer_count_and_density(
    image_input: Union[str, np.ndarray], 
    model: DensityMapRegressor, 
    device: torch.device
) -> Tuple[float, np.ndarray]:
    """Performs inference to predict density map and count."""
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"Could not read image: {image_input}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise TypeError("image_input must be a path (str) or numpy array.")

    processed_image = resize_image(image, size=(224, 224))
    processed_image = normalize_image(processed_image)
    image_tensor = prepare_tensor(processed_image).unsqueeze(0).to(device)

    with torch.no_grad():
        density_map_tensor = model(image_tensor)
    
    density_map = density_map_tensor.squeeze(0).squeeze(0).cpu().numpy()
    predicted_count = float(np.sum(density_map))
    
    return predicted_count, density_map
