import streamlit as st
import sqlite3
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from pathlib import Path
from config import DB_PATH, LABELS as CONFIG_LABELS

st.set_page_config(page_title="Dataset Annotation Dashboard", layout="wide")

# --- DB helpers ---
@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_data(ttl=5)
def load_annotations_df():
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT id, image_path, label, x1, y1, x2, y2, created_at FROM annotations",
        conn
    )
    # Derived columns
    df["width"] = df["x2"] - df["x1"]
    df["height"] = df["y2"] - df["y1"]
    df["area"] = (df["width"] * df["height"]).clip(lower=0)
    df["is_augmented"] = df["image_path"].str.contains(r"(^augmented/|/augmented/)")
    return df

df = load_annotations_df()
if df.empty:
    st.warning("No annotations found.")
    st.stop()

# Dynamic label universe (merge configured + discovered)
all_labels = sorted(set(CONFIG_LABELS).union(df["label"].unique()))

# --- Sidebar controls ---
with st.sidebar:
    st.header("Controls")
    label_filter = st.multiselect("Labels", all_labels, default=all_labels)
    img_origin_filter = st.radio(
        "Image set",
        ["Original only", "Augmented only", "Both"],
        index=2
    )
    heatmap_mode = st.selectbox(
        "Heatmap mode",
        ["Count (box area)", "Normalized", "Centroid", "Per-label (area)"]
    )
    if heatmap_mode == "Per-label (area)":
        heatmap_label = st.selectbox("Heatmap label", all_labels)
    else:
        heatmap_label = None

    show_table = st.checkbox("Show raw annotation table", value=False)
    sort_images_by = st.selectbox(
        "Sort images by",
        ["Path", "Annotation Count", "Newest First", "Oldest First"]
    )
    max_images = st.slider("Max images in selector", 10, 500, 200, step=10)

# --- Filter dataframe ---
mask = df["label"].isin(label_filter)
if img_origin_filter == "Original only":
    mask &= ~df["is_augmented"]
elif img_origin_filter == "Augmented only":
    mask &= df["is_augmented"]
fdf = df[mask].copy()

# --- KPI Row ---
total_images = fdf["image_path"].nunique()
total_ann = len(fdf)
avg_boxes = (total_ann / total_images) if total_images else 0
col1, col2, col3, col4 = st.columns(4)
col1.metric("Images", total_images)
col2.metric("Annotations", total_ann)
col3.metric("Avg boxes/image", f"{avg_boxes:.2f}")
col4.metric("Distinct labels", fdf["label"].nunique())

# --- Label Distribution (Interactive) ---
st.subheader("Label Distribution")
label_counts = (
    fdf.groupby("label")["id"].count().reset_index().rename(columns={"id": "count"})
)
fig_labels = px.bar(
    label_counts,
    x="label",
    y="count",
    color="label",
    title="Annotations per Label",
    height=380
)
fig_labels.update_layout(showlegend=False)
st.plotly_chart(fig_labels, use_container_width=True)

# --- Annotations per Image (Interactive) ---
st.subheader("Annotations per Image")
ann_per_image = fdf.groupby("image_path")["id"].count().reset_index(name="ann_count")
fig_img_hist = px.histogram(
    ann_per_image,
    x="ann_count",
    nbins=20,
    title="Distribution of Annotation Counts per Image",
    height=350
)
st.plotly_chart(fig_img_hist, use_container_width=True)

# --- Image selector preparation ---
# Aggregate counts for sorting
ann_count_map = dict(zip(ann_per_image["image_path"], ann_per_image["ann_count"]))
img_paths = list(ann_count_map.keys())

if sort_images_by == "Annotation Count":
    img_paths.sort(key=lambda p: ann_count_map[p], reverse=True)
elif sort_images_by == "Newest First":
    # approximate by max annotation id for that image
    max_id_map = fdf.groupby("image_path")["id"].max().to_dict()
    img_paths.sort(key=lambda p: max_id_map[p], reverse=True)
elif sort_images_by == "Oldest First":
    min_id_map = fdf.groupby("image_path")["id"].min().to_dict()
    img_paths.sort(key=lambda p: min_id_map[p])
else:
    img_paths.sort()

img_paths = img_paths[:max_images]

st.subheader("Image Browser")
selected_image = st.selectbox("Choose an image", img_paths)

# --- Display selected image with boxes ---
if selected_image:
    raw_img = cv2.imread(selected_image)
    if raw_img is None:
        st.error(f"Could not read {selected_image}")
    else:
        draw = raw_img.copy()
        anns = fdf[fdf["image_path"] == selected_image]

        for _, row in anns.iterrows():
            x1, y1, x2, y2 = int(row.x1), int(row.y1), int(row.x2), int(row.y2)
            label = row.label
            color = (0, 200, 0) if not row.is_augmented else (255, 140, 0)
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                draw,
                label,
                (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1
            )

        if "augmented/" in selected_image:
            suffix = Path(selected_image).stem.split("_")[-1]
            cv2.putText(
                draw,
                f"AUG:{suffix}",
                (5, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        st.image(Image.fromarray(draw_rgb), caption=selected_image, use_container_width=True)

# --- Heatmap Section ---
st.subheader("Spatial Heatmap")
# Determine max canvas
img_dims = {}
for p in fdf["image_path"].unique():
    img = cv2.imread(p)
    if img is not None:
        h, w = img.shape[:2]
        img_dims[p] = (w, h)
if not img_dims:
    st.info("No readable images for heatmap.")
else:
    max_w = max(w for w, _ in img_dims.values())
    max_h = max(h for _, h in img_dims.values())

    heatmap = np.zeros((max_h, max_w), dtype=np.float32)

    hdf = fdf.copy()
    if heatmap_mode == "Per-label (area)" and heatmap_label:
        hdf = hdf[hdf["label"] == heatmap_label]

    for _, row in hdf.iterrows():
        p = row.image_path
        if p not in img_dims:
            continue
        w, h = img_dims[p]
        sx = max_w / w
        sy = max_h / h
        X1 = int(row.x1 * sx)
        Y1 = int(row.y1 * sy)
        X2 = int(row.x2 * sx)
        Y2 = int(row.y2 * sy)
        X1, Y1 = max(0, X1), max(0, Y1)
        X2, Y2 = min(max_w - 1, X2), min(max_h - 1, Y2)
        if X2 <= X1 or Y2 <= Y1:
            continue

        if heatmap_mode in ["Count (box area)", "Per-label (area)"]:
            heatmap[Y1:Y2, X1:X2] += 1.0
        elif heatmap_mode == "Centroid":
            cx = (X1 + X2) // 2
            cy = (Y1 + Y2) // 2
            if 0 <= cy < max_h and 0 <= cx < max_w:
                heatmap[cy, cx] += 1.0
        elif heatmap_mode == "Normalized":
            heatmap[Y1:Y2, X1:X2] += 1.0

    if heatmap_mode == "Normalized" and heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    title_suffix = heatmap_mode if heatmap_mode != "Per-label (area)" else f"Label={heatmap_label}"
    fig_heat = px.imshow(
        heatmap,
        color_continuous_scale="hot",
        origin="upper",
        title=f"Heatmap ({title_suffix})"
    )
    fig_heat.update_layout(margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=True)
    st.plotly_chart(fig_heat, use_container_width=True)

# --- Raw Table (optional) ---
if show_table:
    st.subheader("Filtered Annotations Table")
    st.dataframe(
        fdf.sort_values("id", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=400
    )

st.caption("Dashboard refreshed automatically on interaction.")