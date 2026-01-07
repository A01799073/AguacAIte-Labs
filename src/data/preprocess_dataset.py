import os
import cv2

from pathlib import Path
from typing import Tuple

from src.data.preprocess_functions import preprocess_image

RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")

CLASSES = ["normal","tuberculosis"]

def preprocess_dataset(raw_dir: Path = RAW_PATH, processed_dir: Path= PROCESSED_PATH, size: Tuple[int, int] = (244,244), use_clahe:bool = True):
    """
    Applies the method of processing all images in raw_dir and save it into "processed_dir"
    in order to preserve class foder structures
    """

    for cls in CLASSES:
        input_dir =  raw_dir / cls
        output_dir = processed_dir / cls

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing class: {cls}")

        for img_name in os.listdir(input_dir):
            img_path = input_dir / img_name

            try:
                img = preprocess_image(
                    path=str(img_path),
                    size=size,
                    use_clahe=use_clahe
                )

                # Convert back to uint8 for saving
                img_uint8 = (img * 255).astype("uint8")
                cv2.imwrite(str(output_dir / img_name), img_uint8)

            except Exception as e:
                print(f"[ERROR] {img_name}: {e}")

    print("Dataset preprocessing completed")


if __name__ == "__main__":
    preprocess_dataset()