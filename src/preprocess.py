import cv2
import os
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def preprocess_image(image_path, target_size=(640, 480)):
    """Load, resize, and normalize an image."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read {image_path}")
        return None

    img_resized = cv2.resize(img, target_size)

    img_normalized = cv2.convertScaleAbs(img_resized, alpha=1.2, beta=20)

    return img_normalized

def preprocess_dataset():
    """Process all images from RAW_DIR and save into PROCESSED_DIR."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for img_file in RAW_DIR.iterdir():
        if img_file.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        processed = preprocess_image(img_file)
        if processed is not None:
            save_path = PROCESSED_DIR / img_file.name
            cv2.imwrite(str(save_path), processed)
            print(f"Processed and saved: {save_path}")

if __name__ == "__main__":
    preprocess_dataset()
