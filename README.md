# Autonomous Driving Dataset Preprocessing & Annotation Tool

## 📌 Overview
This project is a **comprehensive pipeline** for preparing, annotating, and visualizing autonomous driving datasets. It integrates **AI-assisted labeling (YOLOv8)**, **data augmentation**, and a **Streamlit dashboard** for analysis, making it easier to build high-quality datasets for computer vision models.

The tool is designed to streamline the entire workflow:
- **Preprocessing** raw datasets (resizing, normalization, cleaning).
- **AI-assisted annotation** using YOLOv8 with manual validation.
- **Augmentation** for dataset diversity.
- **Database storage** with SQLite for structured annotations.
- **Interactive dashboard** for visualization, statistics, and quality checks.

---

## ⚙️ Features
- 🔹 Automated preprocessing and dataset cleaning.
- 🔹 **YOLOv8-powered AI suggestions** for object detection.
- 🔹 **Manual validation loop** for high-quality labels.
- 🔹 Data augmentation (rotation, flipping, color jitter, noise, etc.).
- 🔹 SQLite-based annotation storage with queries.
- 🔹 **Polished Streamlit dashboard** with:
  - Dataset statistics (per-label counts, augmented vs original).
  - Interactive filters.
  - Heatmaps for bounding box density.
  - Image browser with bounding box visualization.
- 🔹 Scalable project structure for future extensions.

---

## 📂 Project Structure
```
Autonomous Driving Dataset Preprocessing & Annotation Tool/
│── data/                # Raw dataset
│── augmented/           # Augmented images
│── notebooks/           # Experimental notebooks
│── src/                 # Source code
│   ├── preprocess.py    # Preprocessing pipeline
│   ├── augment.py       # Data augmentation
│   ├── detector.py      # YOLOv8 detection wrapper
│   ├── annotate.py      # AI-assisted annotation tool
│   ├── db_manager.py    # SQLite database manager
│   ├── dashboard.py     # Streamlit dashboard
│── requirements.txt     # Dependencies
│── README.md            # Project description
```

---

## 🚀 Installation
```bash
# Clone the repository
git clone https://github.com/your-username/autonomous-driving-dataset-tool.git
cd autonomous-driving-dataset-tool

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage
### 1. Preprocess Dataset
```bash
python src/preprocess.py
```

### 2. Run Data Augmentation
```bash
python src/augment.py
```

### 3. AI-Assisted Annotation
```bash
python src/annotate.py
```

### 4. Launch Dashboard
```bash
streamlit run src/dashboard.py
```

---

## 📊 Dashboard Preview
The dashboard provides:
- Dataset statistics per label.
- Augmented vs Original comparison.
- Heatmaps for bounding box density.
- Interactive image explorer with bounding boxes.

---

##  Quantum-Inspired Extension
This project also explores **quantum-inspired optimization techniques** for future extensions in dataset balancing and model training. By leveraging **quantum annealing concepts**, the pipeline could optimize augmentation strategies to achieve better dataset diversity.
