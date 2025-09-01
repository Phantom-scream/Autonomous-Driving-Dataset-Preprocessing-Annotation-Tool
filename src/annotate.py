import cv2
from pathlib import Path
from db_manager import init_db, insert_annotation
from config import PROCESSED_DIR, LABELS
from detector import suggest_boxes

# Globals for drawing boxes
drawing = False
x1, y1, x2, y2 = -1, -1, -1, -1
current_boxes = []  # stores accepted bounding boxes for current image


def draw_box(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing, current_boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        x2, y2 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y
        current_boxes.append(((x1, y1, x2, y2), None))  # None means label not assigned yet


def annotate_image(img_path):
    global x1, y1, x2, y2, current_boxes
    img = cv2.imread(str(img_path))
    clone = img.copy()
    current_boxes = []

    # Load AI-suggested boxes (pass path, not image array)
    try:
        ai_suggestions = suggest_boxes(str(img_path))
    except Exception as e:
        print(f"⚠️ YOLO suggestion failed: {e}")
        ai_suggestions = []

    for (sx1, sy1, sx2, sy2, score, label) in ai_suggestions:
        current_boxes.append(((sx1, sy1, sx2, sy2), label))

    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", draw_box)

    while True:
        display = img.copy()

        # Draw current boxes
        for (coords, label) in current_boxes:
            (bx1, by1, bx2, by2) = coords
            color = (0, 255, 0) if label else (255, 0, 0)
            cv2.rectangle(display, (bx1, by1), (bx2, by2), color, 2)
            if label:
                cv2.putText(display, label, (bx1, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Annotator", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("l"):  # assign label to last drawn/selected box
            if not current_boxes:
                print("⚠️ No box to label.")
                continue
            label = input(f"Enter label {LABELS}: ").strip().lower()
            if label not in LABELS:
                print("⚠️ Invalid label, try again.")
                continue
            coords, _ = current_boxes[-1]
            current_boxes[-1] = (coords, label)
            print(f"✅ Assigned label '{label}'")

        elif key == ord("s"):  # save all labeled boxes
            for (coords, label) in current_boxes:
                if label:
                    (bx1, by1, bx2, by2) = coords
                    insert_annotation(img_path, label, bx1, by1, bx2, by2)
            print(f"💾 Saved annotations for {img_path.name}")
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