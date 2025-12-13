import os
import cv2
from typing import Tuple
from data.preprocess_functions import preprocess_image

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"

CLASSES = ["normal","turberculosis"]

def preprocess_dataset(raw_dir: str = RAW_PATH, processed_dir: str= PROCESSED_PATH, size: Tuple[int, int] = (244,244), use_clahe:bool = True):
    """
    Applies the method of processing all images in raw_dir and save it into "processed_dir"
    in order to preserve class foder structures
    """

    for cls in CLASSES:
        input_dir = os.path.join(raw_dir, cls)
        output_dir = os.path.join(processed_dir, cls)

        os.makedirs(output_dir,exist_ok=True)

        print(f"\nProcessing class: {cls}")

        for filename in os.listdir(input_dir):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir,filename)

            try: 
                image = preprocess_image(path=input_path,size=size, use_clahe=use_clahe)
                # uint8 image (0-255) for compatibility
                image_uint8 = (image * 255).astype("uint8")
                cv2.imwrite(output_path, image_uint8)

            except Exception as e:
                print(f"[ERROR] {filename}: {e}")