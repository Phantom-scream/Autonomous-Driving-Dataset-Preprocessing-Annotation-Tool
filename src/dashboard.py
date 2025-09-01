import streamlit as st
import sqlite3
import cv2
from PIL import Image
from config import DB_PATH, LABELS

# Connect to DB
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

st.title("📊 Dataset Annotation Dashboard")

# Show counts per label
cursor.execute("SELECT label, COUNT(*) FROM annotations GROUP BY label")
rows = cursor.fetchall()
st.subheader("Annotation Counts per Label")
for label in LABELS:
    count = next((c for l, c in rows if l == label), 0)
    st.write(f"**{label}**: {count}")

# Select image to display
st.subheader("View Annotated Images")
cursor.execute("SELECT DISTINCT image_path FROM annotations")
image_paths = [row[0] for row in cursor.fetchall()]

selected_image = st.selectbox("Choose an image", image_paths)

if selected_image:
    img = cv2.imread(selected_image)
    cursor.execute("SELECT x1, y1, x2, y2, label FROM annotations WHERE image_path = ?", (selected_image,))
    annotations = cursor.fetchall()

    # Draw bounding boxes
    for x1, y1, x2, y2, label in annotations:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(Image.fromarray(img_rgb), caption=selected_image, use_container_width=True)