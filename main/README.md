# Lung Cancer Detection & Histopathological Subtyping System
### Final Year Project | Dual-Head CNN with Automated Bounding Box Localization & Web UI

An automated computer-aided diagnostic (CAD) deep learning system that detects lung tumors from CT scan slices, regresses bounding-box spatial coordinates, classifies the carcinoma subtype, and computes multi-class probability confidence scores.

---

## 🚀 Key Features
* **Automated Bounding Box Localization:** Regresses `[xmin, ymin, xmax, ymax]` tumor coordinates on unannotated 2D CT scans.
* **4-Class Histopathology Subtyping:**
  * Adenocarcinoma
  * Squamous Cell Carcinoma
  * Small Cell Carcinoma
  * Large Cell Carcinoma
* **High Diagnostic Performance:** Achieved **92.50% Validation Accuracy**, **92.91% Precision**, and **91.77% F1-score**.
* **Interactive FastAPI Web UI:** Drag-and-drop any `.png`, `.jpg`, or raw `.dcm` (DICOM) scan to get instant visual detection and probability breakdowns.

---

## 🖥️ Interactive Diagnostic Web Dashboard

Below is the interactive FastAPI clinical web application in action, demonstrating real-time CT slice ingestion, automated bounding-box localization, and histopathological classification:

<p align="center">
  <img src="https://github.com/Kgotta-contribute/FINAL_YEAR_PROJECT/blob/main/main/sample_images/UI%20image1.png?raw=true" alt="FastAPI Diagnostic Web UI Dashboard" width="850px">
</p>
<p align="center"><i>Figure 1: Diagnostic Web UI displaying the input unannotated slice alongside the AI-annotated tumor localization, predicted carcinoma subtype, and confidence metrics.</i></p>

<p align="center">
  <img src="https://github.com/Kgotta-contribute/FINAL_YEAR_PROJECT/blob/main/main/sample_images/UI%20Image2.png?raw=true" alt="Model Prediction & Confidence Breakdown" width="850px">
</p>
<p align="center"><i>Figure 2: Real-time inference result highlighting the predicted bounding box, multi-class confidence probability distribution, and sub-100ms latency.</i></p>

---

## 📁 Folder Structure
```text
main/
│
├── app.py                                # FastAPI web server & interactive dashboard
├── train_detector.py                     # Dual-Head CNN architecture & training pipeline
├── dataset_pipeline.py                   # TCIA DICOM & PASCAL-VOC XML annotation parser
├── Lung_Tumor_Detection_Pipeline.ipynb   # Interactive evaluation & visualization notebook
├── best_lung_tumor_model.pth             # Trained model checkpoint weights (27 MB)
├── annotated_dataset.csv                 # Master dataset index with 19,498 annotated slices
├── statistics-clinical-20201221.xlsx     # Clinical and demographic metadata
├── requirements.txt                      # Python dependencies
├── sample_images/                        # Test samples, UI screenshots, and dataset crops
│   ├── UI image1.png                     # Dashboard overview screenshot
│   ├── UI Image2.png                     # Inference result screenshot
│   ├── all_crops.zip                     # Compressed archive of 19,498 tumor crops
│   ├── crops/                            # Curated tumor crop samples (50 per subtype)
│   ├── images/                           # Curated full slice samples (5 per subtype)
│   └── visualizations/                   # 50 pre-drawn ground-truth visualizations
└── .gitignore                            # Prevents accidental dataset upload
```

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
python -m uvicorn app:app --port 8000
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

### 3. (Optional) Retrain or Experiment
```bash
python train_detector.py
```
Or open [`Lung_Tumor_Detection_Pipeline.ipynb`](Lung_Tumor_Detection_Pipeline.ipynb) to inspect evaluation metrics and run visual test comparisons.
