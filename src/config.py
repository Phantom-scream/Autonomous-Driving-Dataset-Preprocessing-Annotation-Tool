from pathlib import Path

# Folders
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Database
DB_PATH = Path("data/annotations.db")

# Preprocessing settings
IMG_TARGET_SIZE = (640, 480)  # width x height
BRIGHTNESS_ALPHA = 1.2  # contrast
BRIGHTNESS_BETA = 20    # brightness

# Annotation labels
LABELS = ["car", "traffic", "pedestrian", "urban"]

# Pipeline settings
MAX_FETCH_ANNOTATIONS = 1000
