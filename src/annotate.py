import cv2
from pathlib import Path
from db_manager import init_db, insert_annotation
from config import PROCESSED_DIR, LABELS

# Globals for drawing boxes
drawing = False
x1, y1, x2, y2 = -1, -1, -1, -1

def draw_box(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        x2, y2 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y

def annotate_image(img_path):
    global x1, y1, x2, y2
    img = cv2.imread(str(img_path))
    clone = img.copy()
    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", draw_box)

    while True:
        display = img.copy()
        if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Annotator", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):  # save annotation
            label = input(f"Enter label {LABELS}: ").strip().lower()
            if label not in LABELS:
                print("⚠️ Invalid label, please try again.")
                continue
            insert_annotation(img_path, label, x1, y1, x2, y2)
            print(f"✅ Saved annotation for {img_path.name}")
            break

        elif key == ord("q"):  # skip image
            print(f"⏭ Skipped {img_path.name}")
            break

    cv2.destroyAllWindows()

def annotate_dataset():
    init_db()
    for img_file in PROCESSED_DIR.iterdir():
        if img_file.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue
        print(f"\n🖼️ Annotating: {img_file.name}")
        annotate_image(img_file)

if __name__ == "__main__":
    annotate_dataset()
