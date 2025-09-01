import sqlite3
from pathlib import Path

DB_PATH = Path("data/annotations.db")

def get_connection():
    """Establish a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)

def init_db():
    """Create the annotations table if it doesn't exist."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                label TEXT NOT NULL,
                x1 INTEGER,
                y1 INTEGER,
                x2 INTEGER,
                y2 INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def insert_annotation(image_path, label, x1, y1, x2, y2):
    """Insert a new annotation into the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO annotations (image_path, label, x1, y1, x2, y2)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (str(image_path), label, x1, y1, x2, y2))
        conn.commit()

def fetch_annotations(limit=10):
    """Fetch annotations (default: 10)."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM annotations LIMIT ?", (limit,))
        return cursor.fetchall()

def fetch_by_image(image_path):
    """Fetch all annotations for a given image."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM annotations WHERE image_path = ?", (str(image_path),))
        return cursor.fetchall()
