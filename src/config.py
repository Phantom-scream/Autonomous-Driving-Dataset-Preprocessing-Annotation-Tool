from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

DB_PATH = Path("data/annotations.db")

IMG_TARGET_SIZE = (640, 480) 
BRIGHTNESS_ALPHA = 1.2 
BRIGHTNESS_BETA = 20    

LABELS = ["car", "traffic", "pedestrian", "urban"]

MAX_FETCH_ANNOTATIONS = 1000
