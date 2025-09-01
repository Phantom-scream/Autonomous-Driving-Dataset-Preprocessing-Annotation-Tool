import cv2
import os
import random
import numpy as np
from pathlib import Path
from db_manager import fetch_all_annotations, insert_annotation
from config import PROCESSED_DIR, DB_PATH

# Augmented images folder
AUGMENTED_DIR = Path("augmented")
AUGMENTED_DIR.mkdir(exist_ok=True)

def rotate_point(x, y, cx, cy, angle, w, h):
    """Rotate a point (x,y) around center (cx,cy) by angle in degrees."""
    angle_rad = np.deg2rad(angle)
    nx = np.cos(angle_rad) * (x - cx) - np.sin(angle_rad) * (y - cy) + cx
    ny = np.sin(angle_rad) * (x - cx) + np.cos(angle_rad) * (y - cy) + cy
    return int(nx), int(ny)

def rotate_bbox(bbox, w, h, angle):
    """Rotate bounding box around image center."""
    x1, y1, x2, y2 = bbox
    cx, cy = w // 2, h // 2
    corners = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
    rotated = [rotate_point(x, y, cx, cy, angle, w, h) for (x,y) in corners]
    xs, ys = zip(*rotated)
    return min(xs), min(ys), max(xs), max(ys)

def flip_bbox(bbox, w, h, mode):
    """Flip bounding box horizontally or vertically."""
    x1, y1, x2, y2 = bbox
    if mode == 1:  
        return w - x2, y1, w - x1, y2
    elif mode == 0: 
        return x1, h - y2, x2, h - y1
    return bbox

def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def flip_image(img, mode):
    return cv2.flip(img, mode)

def adjust_brightness(img, factor=1.0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:,:,2] = hsv[:,:,2] * factor
    hsv[:,:,2][hsv[:,:,2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def augment_and_save(img_path, annotations):
    img = cv2.imread(str(img_path))
    (h, w) = img.shape[:2]
    base_name = img_path.stem

    augmentations = [
        ("rot90", lambda x: rotate_image(x, 90), lambda b: rotate_bbox(b, w, h, 90)),
        ("rot180", lambda x: rotate_image(x, 180), lambda b: rotate_bbox(b, w, h, 180)),
        ("flipH", lambda x: flip_image(x, 1), lambda b: flip_bbox(b, w, h, 1)),
        ("flipV", lambda x: flip_image(x, 0), lambda b: flip_bbox(b, w, h, 0)),
        ("bright", lambda x: adjust_brightness(x, random.uniform(0.5, 1.5)), lambda b: b),
    ]

    for aug_name, img_fn, bbox_fn in augmentations:
        aug_img = img_fn(img)
        aug_file = AUGMENTED_DIR / f"{base_name}_{aug_name}.jpg"
        cv2.imwrite(str(aug_file), aug_img)

        # Save transformed annotations
        for ann in annotations:
            bbox = (ann["x1"], ann["y1"], ann["x2"], ann["y2"])
            new_bbox = bbox_fn(bbox)
            insert_annotation(
                str(aug_file),
                ann["label"],
                new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3]
            )
        print(f"✅ Augmented saved: {aug_file.name}")

def augment_dataset():
    annotations = fetch_all_annotations()
    grouped = {}

    for ann in annotations:
        img_path = Path(ann["image_path"])
        if img_path not in grouped:
            grouped[img_path] = []
        grouped[img_path].append(ann)

    print(f"🔄 Found {len(grouped)} images to augment.")

    for img_path, anns in grouped.items():
        if not img_path.exists():
            continue
        augment_and_save(img_path, anns)

if __name__ == "__main__":
    augment_dataset()