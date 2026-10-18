# Lung Cancer Detection & Histopathological Subtyping System
### Final Year Project | Dual-Head CNN with Automated Bounding Box Localization & Web UI

An automated diagnostic deep learning system that detects lung tumors from CT scan slices, draws bounding boxes, classifies the carcinoma type, and computes the probability confidence score.

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

## 📁 Folder Structure
```text
main/
│
├── app.py                                # FastAPI web server & interactive dashboard
├── train_detector.py                     # Dual-Head CNN architecture & training pipeline
├── dataset_pipeline.py                   # TCIA DICOM & PASCAL-VOC XML annotation parser
├── Lung_Tumor_Detection_Pipeline.ipynb   # Interactive evaluation & visualization notebook
├── best_lung_tumor_model.pth             # Trained model checkpoint weights (27 MB)
├── requirements.txt                      # Python dependencies
├── sample_images/                        # Test samples for instant 1-click evaluation
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
