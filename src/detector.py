from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def suggest_boxes(image_path):
    """
    Run YOLOv8 on an image or image path and return suggested bounding boxes.
    Returns list of tuples: (x1, y1, x2, y2, score, label)
    """
    results = model(image_path)
    suggestions = []
    for b in results[0].boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        cls_id = int(b.cls[0])
        label = model.names[cls_id]
        score = float(b.conf[0])
        suggestions.append((x1, y1, x2, y2, score, label))
    return suggestions
