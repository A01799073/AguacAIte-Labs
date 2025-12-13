# Libraries
import cv2
import numpy as np
import os

# Loading images in graysacalde format
def load_images(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image

# Spatial Standardization
def resize_image(image, size=(244, 244))-> np.ndarray:
    # Resize awhile preserving structure
    return cv2.resize(image,size, interpolation = cv2.INTER_AREA)

# Normalization
def normalize_image(image: np.ndarray)-> np.ndarray:
    image=image.astype(np.float32)/255.0
    return image

# Contrar Enhacement (CLAHE)
def apply_clahe(image : np.ndarray,clip_limit = 2.0, grid_size=(8,8)) -> np.ndarray:
    # Enhace image constrar using CLAHE, recommended for X-ray.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize = grid_size)
    return clahe.apply(image)

# Full Pipeline
def preprocess_image(path : str, size=(244,244), use_clahe: bool=True)-> np.ndarray:
    # Load -> resize -> (CLAHE) -> normalize
    image = load_images(path)
    image = resize_image(image,size)

    if use_clahe:
        image = apply_clahe(image)
    
    image = normalize_image(image)

    return image