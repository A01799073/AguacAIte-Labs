import cv2
import numpy as np

# Loading images in graysacalde format
def load_images(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image

# Spatial Standardization
def resize_image(image, size =(244, 244))-> np.ndarray:
    # Resize awhile preserving structure
    return cv2.resize(image,size, interpolation = cv2.INTER_AREA)

# Normalization
def normalize_image(image: np.ndarray)-> np.ndarray:
    image=image.astype(np.float32)/255.0
    return image

# Contrast Enhacement (CLAHE)
def apply_clahe(image : np.ndarray,clip_limit = 3.0, grid_size = (8,8)) -> np.ndarray:
    # Enhace image constrar using CLAHE, recommended for X-ray.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize = grid_size)
    return clahe.apply(image)

# Gamma correction (enhance dark regions)
def gamma_correction(image: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    image = image.astype(np.float32)/255
    image = np.power(image,gamma)
    return (image * 255).astype("uint8")

# Full Pipeline
def preprocess_image(path : str, size = (244,244), use_clahe: bool = True, use_gamma: bool = True)-> np.ndarray:
    # Load -> resize -> (CLAHE) -> Gamma correction -> normalize
    image = load_images(path)
    image = resize_image(image,size)

    if use_clahe:
        image = apply_clahe(image)

    if use_gamma:
        image = gamma_correction(image)
    
    image = normalize_image(image)

    return image