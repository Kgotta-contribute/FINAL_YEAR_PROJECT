# Automated Lung Carcinoma Detection & Histopathological Subtyping Using Deep Convolutional Neural Networks

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![License-MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status-Active](https://img.shields.io/badge/Status-Complete%20%2F%20Production%20Ready-success.svg)](#)

---

## 📌 Executive Summary

Lung cancer remains one of the leading causes of oncological mortality worldwide, where early detection and precise histological subtyping significantly improve clinical prognosis. 

This undergraduate engineering capstone project introduces an end-to-end Computer-Aided Diagnostic (CAD) system utilizing deep Convolutional Neural Networks (CNNs). The pipeline processes volumetric Computed Tomography (CT) scans, identifies pulmonary lesions, regresses bounding-box spatial coordinates $(x_{min}, y_{min}, x_{max}, y_{max})$, and classifies malignant findings into four primary histopathological subtypes:
1. **Adenocarcinoma (ADC)**
2. **Squamous Cell Carcinoma (SCC)**
3. **Small Cell Carcinoma (SCLC)**
4. **Large Cell Carcinoma (LCLC)**

The system delivers real-time diagnostic inference through an interactive **FastAPI** web dashboard.

---

## 🔬 Radiological Morphology & Lesion Analysis

Pulmonary CT imaging displays distinct attenuation patterns that indicate neoplastic progression. Below are representative axial CT slices demonstrating malignant pulmonary nodules versus healthy parenchyma.

### Positive Malignant Slices (Pulmonary Neoplasm Present)
<p align="center">
  <img src="https://github.com/Kgotta-contribute/FINAL_YEAR_PROJECT/blob/main/Images/cropped_130.png?raw=true" alt="Cropped Malignant Region" width="380px" style="margin-right: 20px;">
  <img src="https://github.com/Kgotta-contribute/FINAL_YEAR_PROJECT/blob/main/Images/130.png?raw=true" alt="Full Axial Slice with Lesion" width="380px">
</p>
<p align="center"><i>Figure 1: Axial CT slice and localized region-of-interest (ROI) exhibiting soft-tissue attenuation, spiculation, and architectural distortion characteristic of primary lung carcinoma.</i></p>

### Normal Control Slices (Healthy Pulmonary Parenchyma)
<p align="center">
  <img src="https://github.com/Kgotta-contribute/FINAL_YEAR_PROJECT/blob/main/Images/116.png?raw=true" alt="Normal Lung Parenchyma Slice 1" width="380px" style="margin-right: 20px;">
  <img src="https://github.com/Kgotta-contribute/FINAL_YEAR_PROJECT/blob/main/Images/110.png?raw=true" alt="Normal Lung Parenchyma Slice 2" width="380px">
</p>
<p align="center"><i>Figure 2: Control axial CT scans displaying uniform parenchymal aeration and normal vascular arborization without focal consolidations or nodules.</i></p>

---

## 🔄 Data Augmentation & Invariance Pipeline

To mitigate overfitting and address class imbalance inherent in medical imaging datasets, balanced affine and photometric transformations were introduced.

### Control Set (Non-Augmented Normal Slices)
<p align="center">
  <img src="https://raw.githubusercontent.com/Kgotta-contribute/FINAL_YEAR_PROJECT/9ed3017973b20d5f0fa6df1fb6e38c7514088035/Images/No_AUG.png" alt="Non-Augmented Control Class" width="700px">
</p>
<p align="center"><i>Figure 3: Representative sample grid of non-augmented normal pulmonary slices.</i></p>

### Augmented Malignant Set (Transformed Positive Slices)
<p align="center">
  <img src="https://raw.githubusercontent.com/Kgotta-contribute/FINAL_YEAR_PROJECT/9ed3017973b20d5f0fa6df1fb6e38c7514088035/Images/YES_AUG.jpg" alt="Augmented Positive Carcinoma Class" width="700px">
</p>
<p align="center"><i>Figure 4: Multi-slice affine transformations (horizontal flips with synchronized coordinate transforms, intensity normalization, and contrast adjustments) implemented to double effective training volume.</i></p>

---

## 🖥️ Volumetric Multi-Planar Reconstruction (MPR) via RadiAnt DICOM

Raw multi-detector helical CT studies were evaluated across axial, coronal, and sagittal reconstructions using calibrated Hounsfield Unit (HU) lung window settings ($\text{Window Width} = 1400\text{--}1600\,\text{HU}$, $\text{Window Level} = -600\text{--}-700\,\text{HU}$).

### Volumetric Parenchymal Assessment
<p align="center">
  <img src="https://raw.githubusercontent.com/Kgotta-contribute/FINAL_YEAR_PROJECT/9ed3017973b20d5f0fa6df1fb6e38c7514088035/Images/Screenshot%20(182).png" alt="Axial Lung Window Overview" width="620px">
</p>
<p align="center"><i>Figure 5: High-resolution axial volumetric reconstruction illustrating tracheobronchial branching and bilateral lung field aeration.</i></p>

### Pathological Findings: Ground-Glass Opacities & Hyperdense Nodules
Infiltrative tumor margins, focal ground-glass opacities (GGO), and pleural tags indicate neoplastic infiltration into adjacent parenchymal lobules:

<p align="center">
  <img src="https://raw.githubusercontent.com/Kgotta-contribute/FINAL_YEAR_PROJECT/9ed3017973b20d5f0fa6df1fb6e38c7514088035/Images/Screenshot%20(183).png" alt="Pathological Infiltration Analysis" width="600px" style="margin-right: 20px;">
  <img src="https://raw.githubusercontent.com/Kgotta-contribute/FINAL_YEAR_PROJECT/9ed3017973b20d5f0fa6df1fb6e38c7514088035/Images/Screenshot%20(184).png" alt="Lobular Mass Localization" width="600px">
</p>
<p align="center"><i>Figure 6: High-attenuation focal masses demonstrating lobulation, heterogeneous internal architecture, and peripheral vascular convergence.</i></p>

### Mediastinal & Thoracic Visceral Relationships
<p align="center">
  <img src="https://raw.githubusercontent.com/Kgotta-contribute/FINAL_YEAR_PROJECT/9ed3017973b20d5f0fa6df1fb6e38c7514088035/Images/Screenshot%20(185).png" alt="Mediastinal Window Reconstruction" width="600px" style="margin-right: 20px;">
  <img src="https://raw.githubusercontent.com/Kgotta-contribute/FINAL_YEAR_PROJECT/9ed3017973b20d5f0fa6df1fb6e38c7514088035/Images/Screenshot%20(186).png" alt="Visceral Anatomical Context" width="600px">
</p>
<p align="center"><i>Figure 7: Soft-tissue mediastinal windowing verifying relationships with the cardiovascular silhouette, hilar lymph nodes, and pleural boundaries.</i></p>

---

## 📈 Model Optimization & Training Diagnostics

The model optimization progression demonstrates loss attenuation and metric convergence across successive epochs:

<p align="center">
  <img src="https://raw.githubusercontent.com/Kgotta-contribute/FINAL_YEAR_PROJECT/9ed3017973b20d5f0fa6df1fb6e38c7514088035/Images/EPOCH.png" alt="Initial Training Loss Curves" width="700px">
</p>
<p align="center"><i>Figure 8: Baseline training trajectories monitoring cross-entropy minimization and convergence behavior.</i></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Kgotta-contribute/FINAL_YEAR_PROJECT/9ed3017973b20d5f0fa6df1fb6e38c7514088035/Images/EPOCH2.png" alt="Intermediate Metric Progression" width="700px">
</p>
<p align="center"><i>Figure 9: Regularization tuning and learning rate adjustments to eliminate validation loss divergence.</i></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Kgotta-contribute/FINAL_YEAR_PROJECT/9ed3017973b20d5f0fa6df1fb6e38c7514088035/Images/EPOCH_res.png" alt="Convergence & Final Metrics" width="700px">
</p>
<p align="center"><i>Figure 10: Validation performance stabilization achieving optimal classification accuracy and generalized representation.</i></p>

---

## 🏗️ System Architecture & Production Deployment (`/main`)

The project has been refactored into a modular, production-ready directory [`main/`](main/):

| Component | File Path | Functionality |
| :--- | :--- | :--- |
| **Interactive Clinical UI** | [`main/app.py`](main/app.py) | Full-stack FastAPI application serving real-time bounding box visualization, multi-class probability scores, and native DICOM (`.dcm`) parsing. |
| **Dual-Head CNN Architecture** | [`main/train_detector.py`](main/train_detector.py) | PyTorch model uniting spatial bounding box regression (Smooth L1 Loss) with multi-class carcinoma classification (Cross-Entropy Loss). |
| **Dataset Ingestion Pipeline** | [`main/dataset_pipeline.py`](main/dataset_pipeline.py) | Multi-threaded extraction connecting PASCAL VOC XML annotations to volumetric DICOM scans with Hounsfield Unit lung windowing. |
| **Trained Checkpoint** | [`main/best_lung_tumor_model.pth`](main/best_lung_tumor_model.pth) | Optimized PyTorch model weights achieving **92.50% Validation Accuracy**. |
| **Interactive Notebook** | [`main/Lung_Tumor_Detection_Pipeline.ipynb`](main/Lung_Tumor_Detection_Pipeline.ipynb) | End-to-end evaluation notebook demonstrating side-by-side detection on unannotated CT scans. |

### Final Benchmark Performance

| Evaluation Metric | Score Achieved |
| :--- | :--- |
| **Validation Accuracy** | **`92.50%`** |
| **Validation Precision** | **`92.91%`** |
| **Validation Recall (Sensitivity)** | **`92.50%`** |
| **Validation F1-Score** | **`91.77%`** |
| **Inference Latency** | **`< 60 ms` per slice** |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Kgotta-contribute/FINAL_YEAR_PROJECT.git
cd FINAL_YEAR_PROJECT/main
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Interactive Diagnostic Dashboard
```bash
python -m uvicorn app:app --port 8000
```
Open your browser and visit:
👉 **`http://127.0.0.1:8000`**

Drag and drop any unannotated CT scan slice (PNG, JPG, or raw DICOM `.dcm`) to obtain real-time tumor localization and histopathological classification.

---

## 👨‍💻 Author & Acknowledgements
* **Author:** [Chhavi Verma (Kgotta-contribute)](https://github.com/Kgotta-contribute)
* **Dataset Reference:** The Cancer Imaging Archive (TCIA) *Lung-PET-CT-Dx* Collection (Li et al., 2020).
