from preprocess import preprocess_dataset
from annotate import annotate_dataset
from db_manager import fetch_annotations
import matplotlib.pyplot as plt
import cv2

def run_pipeline():
    print("Starting dataset pipeline...")

    print("\n Step 1: Preprocessing images...")
    preprocess_dataset()

    print("\n Step 2: Annotating images...")
    annotate_dataset()

    print("\n Step 3: Dataset summary:")
    annotations = fetch_annotations(limit=1000) 
    label_counts = {}
    for ann in annotations:
        label = ann[2] 
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\n Annotation counts per label:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    if annotations:
        sample = annotations[0]
        img_path = sample[1]
        x1, y1, x2, y2 = sample[3], sample[4], sample[5], sample[6]
        img = cv2.imread(img_path)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Sample Annotation", img)
        print("\nPress any key on the image window to finish...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()
