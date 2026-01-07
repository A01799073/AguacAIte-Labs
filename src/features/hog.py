import numpy as np

from skimage.feature import hog
from skimage.color import rgb2gray


def extract_hog_features(image, pixels_per_cell=(16, 16),cells_per_block = (2, 2), orientations = 9):
    """
    Compute HOG features for a single image.
    """
    
    if image.ndim == 3:
        image = rgb2gray(image)

    return hog(
        image,
        orientations = orientations,
        pixels_per_cell = pixels_per_cell,
        cells_per_block = cells_per_block,
        block_norm = "L2-Hys",
        feature_vector = True
    )


def extract_hog_dataset(images, **hog_params):
    """
    Compute HOG features for a collection of images.
    """

    features = []
    for img in images:
        features.append(extract_hog_features(img, **hog_params))

    return np.array(features)
