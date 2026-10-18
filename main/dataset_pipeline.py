"""
Automated Pipeline for Lung-PET-CT-Dx Dataset
1. Maps XML PASCAL-VOC bounding box annotations to DICOM CT slices via SOPInstanceUID.
2. Extracts and normalizes CT slices to 8-bit PNG images using standard Lung Windowing (WL: -600, WW: 1500).
3. Generates cropped tumor patches for CNN classification models.
4. Produces sample bounding box overlay visualizations.
5. Generates 'annotated_dataset.csv' index.
"""

import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

# --- Configuration Paths ---
BASE_DIR = r"C:\Users\verma\Downloads\Lung Pet CT scan"
ANNOTATION_ROOT = os.path.join(BASE_DIR, "Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020", "Annotation")
IMAGE_ROOT = os.path.join(BASE_DIR, "Lung Images", "lung_pet_ct_dx")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_png")

IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
CROPS_DIR = os.path.join(OUTPUT_DIR, "crops")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
CSV_PATH = os.path.join(OUTPUT_DIR, "annotated_dataset.csv")

CLASS_MAPPING = {
    "A": "Adenocarcinoma",
    "B": "Small Cell Carcinoma",
    "E": "Large Cell Carcinoma",
    "G": "Squamous Cell Carcinoma"
}

def apply_lung_window(pixel_array, slope, intercept, center=-600, width=1500):
    """Convert raw DICOM to Hounsfield Units (HU) and apply Lung Window normalization to 0-255."""
    hu = pixel_array.astype(np.float32) * float(slope) + float(intercept)
    lower = center - width / 2.0
    upper = center + width / 2.0
    windowed = np.clip(hu, lower, upper)
    norm = ((windowed - lower) / (upper - lower) * 255.0).astype(np.uint8)
    return norm

def scan_patient(patient_id):
    """Finds all DICOM files matching XML annotations for a given patient."""
    xml_dir = os.path.join(ANNOTATION_ROOT, patient_id)
    if not os.path.isdir(xml_dir):
        return []

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith(".xml")]
    if not xml_files:
        return []

    # Map of SOPInstanceUID -> list of annotations in that slice
    slice_annotations = {}
    for xf in xml_files:
        sop_uid = os.path.splitext(xf)[0]
        xml_path = os.path.join(xml_dir, xf)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            boxes = []
            for obj in root.findall("object"):
                cls_code = obj.find("name").text.strip().upper()
                bndbox = obj.find("bndbox")
                xmin = max(0, int(float(bndbox.find("xmin").text)))
                ymin = max(0, int(float(bndbox.find("ymin").text)))
                xmax = min(512, int(float(bndbox.find("xmax").text)))
                ymax = min(512, int(float(bndbox.find("ymax").text)))
                boxes.append({
                    "class_code": cls_code,
                    "class_name": CLASS_MAPPING.get(cls_code, "Unknown"),
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                })
            if boxes:
                slice_annotations[sop_uid] = (xml_path, boxes)
        except Exception:
            continue

    if not slice_annotations:
        return []

    # Find matching DICOM files in lung_pet_ct_dx
    target_uids = set(slice_annotations.keys())
    matched_tasks = []
    patient_dcm_dir = os.path.join(IMAGE_ROOT, f"Lung_Dx-{patient_id}")
    if not os.path.exists(patient_dcm_dir):
        return []

    for root, _, files in os.walk(patient_dcm_dir):
        for f in files:
            if f.endswith(".dcm"):
                dcm_path = os.path.join(root, f)
                try:
                    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                    uid = getattr(ds, "SOPInstanceUID", None)
                    if uid in target_uids:
                        xml_path, boxes = slice_annotations[uid]
                        matched_tasks.append({
                            "patient_id": patient_id,
                            "sop_uid": uid,
                            "dcm_path": dcm_path,
                            "xml_path": xml_path,
                            "boxes": boxes
                        })
                except Exception:
                    continue

    return matched_tasks

def process_and_save_slice(task, save_vis=False):
    """Reads DICOM pixel data, applies windowing, and saves full PNG + tumor crops."""
    patient_id = task["patient_id"]
    sop_uid = task["sop_uid"]
    dcm_path = task["dcm_path"]
    boxes = task["boxes"]

    full_png_filename = f"{patient_id}_{sop_uid}.png"
    full_png_path = os.path.join(IMAGES_DIR, full_png_filename)

    records = []

    # Check if already processed
    need_read = not os.path.exists(full_png_path)
    for i in range(len(boxes)):
        crop_filename = f"{patient_id}_{sop_uid}_crop_{i}.png"
        crop_path = os.path.join(CROPS_DIR, crop_filename)
        if not os.path.exists(crop_path):
            need_read = True

    if need_read:
        try:
            ds = pydicom.dcmread(dcm_path)
            slope = getattr(ds, "RescaleSlope", 1)
            intercept = getattr(ds, "RescaleIntercept", 0)
            img = apply_lung_window(ds.pixel_array, slope, intercept)
            cv2.imwrite(full_png_path, img)

            # Generate crops
            h, w = img.shape
            for i, box in enumerate(boxes):
                crop_filename = f"{patient_id}_{sop_uid}_crop_{i}.png"
                crop_path = os.path.join(CROPS_DIR, crop_filename)
                
                # Safe crop with boundary checking
                ymin = max(0, min(box["ymin"], h - 1))
                ymax = max(ymin + 1, min(box["ymax"], h))
                xmin = max(0, min(box["xmin"], w - 1))
                xmax = max(xmin + 1, min(box["xmax"], w))

                crop_img = img[ymin:ymax, xmin:xmax]
                if crop_img.size > 0:
                    cv2.imwrite(crop_path, crop_img)

            # Optional visualization
            if save_vis:
                vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for box in boxes:
                    cv2.rectangle(vis_img, (box["xmin"], box["ymin"]), (box["xmax"], box["ymax"]), (0, 0, 255), 2)
                    cv2.putText(vis_img, f"{box['class_name']}", (box["xmin"], max(15, box["ymin"] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                vis_path = os.path.join(VIS_DIR, f"{patient_id}_{sop_uid}_vis.png")
                cv2.imwrite(vis_path, vis_img)

        except Exception as e:
            return []

    # Build CSV rows
    for i, box in enumerate(boxes):
        crop_filename = f"{patient_id}_{sop_uid}_crop_{i}.png"
        crop_path = os.path.join(CROPS_DIR, crop_filename)
        records.append({
            "patient_id": patient_id,
            "sop_uid": sop_uid,
            "class_code": box["class_code"],
            "class_name": box["class_name"],
            "xmin": box["xmin"],
            "ymin": box["ymin"],
            "xmax": box["xmax"],
            "ymax": box["ymax"],
            "box_width": box["xmax"] - box["xmin"],
            "box_height": box["ymax"] - box["ymin"],
            "full_png_path": os.path.abspath(full_png_path),
            "crop_png_path": os.path.abspath(crop_path),
            "dcm_path": os.path.abspath(dcm_path)
        })

    return records

def main():
    print("=" * 70)
    print("Lung-PET-CT-Dx Automated Annotation & PNG Extraction Pipeline")
    print("=" * 70)

    # 1. Create clean output directories
    for directory in [OUTPUT_DIR, IMAGES_DIR, CROPS_DIR, VIS_DIR]:
        os.makedirs(directory, exist_ok=True)

    # 2. Get list of patients
    patients = sorted([d for d in os.listdir(ANNOTATION_ROOT) if os.path.isdir(os.path.join(ANNOTATION_ROOT, d))])
    print(f"[*] Found {len(patients)} patient annotation folders.")
    print("[*] Phase 1: Scanning DICOM files and matching annotations...")

    all_tasks = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(scan_patient, pid): pid for pid in patients}
        for future in tqdm(as_completed(futures), total=len(patients), desc="Scanning patients"):
            res = future.result()
            if res:
                all_tasks.extend(res)

    print(f"\n[+] Matched {len(all_tasks)} annotated DICOM slices across all patients!")

    # 3. Process slices & save PNGs
    print("\n[*] Phase 2: Converting DICOMs to Lung-windowed PNGs & extracting tumor crops...")
    all_records = []
    
    # Save visualizations for first 50 slices
    save_vis_limit = 50
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(process_and_save_slice, task, idx < save_vis_limit): idx
            for idx, task in enumerate(all_tasks)
        }
        for future in tqdm(as_completed(futures), total=len(all_tasks), desc="Exporting PNGs"):
            res = future.result()
            if res:
                all_records.extend(res)

    # 4. Save metadata DataFrame
    df = pd.DataFrame(all_records)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n[+] Successfully saved {len(df)} annotation records to:")
    print(f"    {CSV_PATH}")

    # 5. Print Summary Statistics
    print("\n" + "=" * 50)
    print("DATASET EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Total Unique Slices Processed: {df['sop_uid'].nunique()}")
    print(f"Total Tumor Crops Generated:   {len(df)}")
    print(f"Total Patients Represented:    {df['patient_id'].nunique()}")
    print("\nClass Distribution:")
    class_counts = df["class_name"].value_counts()
    for cname, count in class_counts.items():
        print(f"  - {cname:25s}: {count:5d} ({count / len(df) * 100:.1f}%)")
    print("=" * 50)
    print("Output folders created inside C:\\Users\\verma\\Downloads\\Lung Pet CT scan\\dataset_png:")
    print(f"  - images/:         Full 2D slice PNGs (512x512)")
    print(f"  - crops/:          Cropped tumor patches (ready for CNN input)")
    print(f"  - visualizations/: Sample bounding-box overlay PNGs")
    print(f"  - annotated_dataset.csv: Full metadata index")
    print("=" * 50)

if __name__ == "__main__":
    main()
