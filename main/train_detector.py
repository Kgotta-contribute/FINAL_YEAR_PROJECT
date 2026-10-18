"""
End-to-End Lung Tumor Detection & Classification Pipeline
1. Loads annotated dataset from 'dataset_png/annotated_dataset.csv'.
2. Performs Data Augmentation (Horizontal flips with bounding-box adjustment) to double the data.
3. Implements a Dual-Head CNN Architecture:
   - Head 1 (Bounding Box Regressor): Predicts [xmin, ymin, xmax, ymax] normalized coordinates.
   - Head 2 (Classifier): Predicts probability distribution across carcinoma types.
4. Computes Accuracy, Precision, Recall, F1-Score, and Mean IoU (Intersection-over-Union).
5. Saves the trained model weights to 'best_lung_tumor_model.pth'.
6. Includes an inference function for FastAPI integration.
"""

import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- Configurations ---
CSV_PATH = r"C:\Users\verma\Downloads\Lung Pet CT scan\dataset_png\annotated_dataset.csv"
MODEL_SAVE_PATH = r"C:\Users\verma\Downloads\Lung Pet CT scan\best_lung_tumor_model.pth"
IMG_SIZE = 224  # Standard CNN resolution for balanced speed and feature retention
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Adenocarcinoma", "Squamous Cell Carcinoma", "Small Cell Carcinoma", "Large Cell Carcinoma"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# --- 1. Dataset & Augmentation ---
class LungTumorDataset(Dataset):
    def __init__(self, df, img_size=IMG_SIZE, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        # If augment is True, double the dataset amount
        return len(self.df) * 2 if self.augment else len(self.df)

    def __getitem__(self, idx):
        is_augmented = False
        if self.augment and idx >= len(self.df):
            idx = idx - len(self.df)
            is_augmented = True

        row = self.df.iloc[idx]
        img_path = row["full_png_path"]

        # Read grayscale 8-bit image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback blank image if file read issue
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            h, w = self.img_size, self.img_size
        else:
            h, w = img.shape

        # Normalize bounding box coordinates to [0, 1] relative to original image size
        xmin = float(row["xmin"]) / w
        ymin = float(row["ymin"]) / h
        xmax = float(row["xmax"]) / w
        ymax = float(row["ymax"]) / h

        # Apply Augmentation (Horizontal Flip + Box Coordinate Flip)
        if is_augmented:
            img = cv2.flip(img, 1)  # horizontal flip
            new_xmin = 1.0 - xmax
            new_xmax = 1.0 - xmin
            xmin, xmax = new_xmin, new_xmax

            # Random slight brightness variation
            factor = random.uniform(0.85, 1.15)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        # Resize image to model input size
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert to Tensor (1, H, W) normalized to [0, 1]
        img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0) / 255.0
        
        # Bounding box tensor [xmin, ymin, xmax, ymax]
        bbox_tensor = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)

        # Class label
        cls_idx = CLASS_TO_IDX.get(row["class_name"], 0)
        label_tensor = torch.tensor(cls_idx, dtype=torch.long)

        return img_tensor, bbox_tensor, label_tensor

# --- 2. Dual-Head CNN Model Architecture ---
class DualHeadLungCNN(nn.Module):
    """
    Dual-Head CNN:
    - Backbone: 4 Conv-BatchNorm-ReLU-MaxPool blocks.
    - Head 1 (Box Regressor): Outputs 4 normalized coordinates [xmin, ymin, xmax, ymax] via Sigmoid.
    - Head 2 (Classifier): Outputs logits across the 4 carcinoma types.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super(DualHeadLungCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Adaptive pool to 7x7
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        feature_dim = 256 * 7 * 7

        # Head 1: Bounding Box Regressor (xmin, ymin, xmax, ymax)
        self.box_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 4),
            nn.Sigmoid()  # Keeps outputs in range [0, 1]
        )

        # Head 2: Carcinoma Type Classifier
        self.cls_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.features(x)
        flattened = feats.view(feats.size(0), -1)
        pred_boxes = self.box_head(flattened)
        pred_logits = self.cls_head(flattened)
        return pred_boxes, pred_logits

# --- 3. Evaluation Metrics ---
def compute_iou(boxA, boxB):
    """Computes Intersection over Union (IoU) between two bounding boxes [xmin, ymin, xmax, ymax]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    boxA_area = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxB_area = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union_area = boxA_area + boxB_area - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

# --- 4. Training & Validation Loop ---
def train_model():
    print("=" * 70)
    print("Training Dual-Head Lung Tumor Detection & Classification CNN")
    print(f"Device: {DEVICE}")
    print("=" * 70)

    # Load master metadata
    df = pd.read_csv(CSV_PATH)
    # Filter valid existing files
    df = df[df["class_name"].isin(CLASS_NAMES)].reset_index(drop=True)

    # For fast & balanced demonstration, sample up to 2000 images per class or overall balanced subset
    # (can use full df for multi-hour training)
    sample_df = df.sample(n=min(3000, len(df)), random_state=42).reset_index(drop=True)
    print(f"Using balanced sample of {len(sample_df)} annotated slices for training.")

    train_df, val_df = train_test_split(sample_df, test_size=0.2, random_state=42, stratify=sample_df["class_name"])
    
    # Dataset with 2x augmentation on training set
    train_dataset = LungTumorDataset(train_df, augment=True)
    val_dataset = LungTumorDataset(val_df, augment=False)

    print(f"Train samples (with 2x augmentation): {len(train_dataset)}")
    print(f"Validation samples:                    {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DualHeadLungCNN(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Losses
    box_criterion = nn.SmoothL1Loss()  # Huber loss for bounding boxes
    cls_criterion = nn.CrossEntropyLoss()  # Classification loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # --- Training ---
        model.train()
        train_box_loss = 0.0
        train_cls_loss = 0.0

        for imgs, true_boxes, true_labels in train_loader:
            imgs = imgs.to(DEVICE)
            true_boxes = true_boxes.to(DEVICE)
            true_labels = true_labels.to(DEVICE)

            optimizer.zero_grad()
            pred_boxes, pred_logits = model(imgs)

            loss_box = box_criterion(pred_boxes, true_boxes)
            loss_cls = cls_criterion(pred_logits, true_labels)
            total_loss = (10.0 * loss_box) + loss_cls  # Balance box regression and class loss

            total_loss.backward()
            optimizer.step()

            train_box_loss += loss_box.item() * imgs.size(0)
            train_cls_loss += loss_cls.item() * imgs.size(0)

        train_box_loss /= len(train_dataset)
        train_cls_loss /= len(train_dataset)

        # --- Validation ---
        model.eval()
        val_box_loss = 0.0
        val_cls_loss = 0.0
        all_preds = []
        all_trues = []
        all_ious = []

        with torch.no_grad():
            for imgs, true_boxes, true_labels in val_loader:
                imgs = imgs.to(DEVICE)
                true_boxes = true_boxes.to(DEVICE)
                true_labels = true_labels.to(DEVICE)

                pred_boxes, pred_logits = model(imgs)

                loss_box = box_criterion(pred_boxes, true_boxes)
                loss_cls = cls_criterion(pred_logits, true_labels)

                val_box_loss += loss_box.item() * imgs.size(0)
                val_cls_loss += loss_cls.item() * imgs.size(0)

                preds = torch.argmax(pred_logits, dim=1).cpu().numpy()
                trues = true_labels.cpu().numpy()
                all_preds.extend(preds)
                all_trues.extend(trues)

                # Compute IoU for bounding box accuracy
                p_boxes = pred_boxes.cpu().numpy()
                t_boxes = true_boxes.cpu().numpy()
                for pb, tb in zip(p_boxes, t_boxes):
                    all_ious.append(compute_iou(pb, tb))

        val_box_loss /= len(val_dataset)
        val_cls_loss /= len(val_dataset)
        total_val_loss = (10.0 * val_box_loss) + val_cls_loss

        # Classification Metrics
        acc = accuracy_score(all_trues, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_trues, all_preds, average="weighted", zero_division=0)
        mean_iou = np.mean(all_ious)

        print(f"\n--- Epoch [{epoch}/{EPOCHS}] ---")
        print(f"  Train Loss: Box={train_box_loss:.4f}, Cls={train_cls_loss:.4f}")
        print(f"  Val Loss:   Box={val_box_loss:.4f}, Cls={val_cls_loss:.4f} (Total={total_val_loss:.4f})")
        print(f"  [Metrics] Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}% | Mean IoU: {mean_iou:.3f}")

        # Save Best Model Checkpoint
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "img_size": IMG_SIZE,
                "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "mean_iou": mean_iou}
            }, MODEL_SAVE_PATH)
            print(f"  --> Saved new best checkpoint to {MODEL_SAVE_PATH}")

    print("\n" + "=" * 70)
    print(f"Training Complete! Best model saved to: {MODEL_SAVE_PATH}")
    print("=" * 70)

# --- 5. Standalone Inference Function for FastAPI / UI ---
def predict_image(image_path, model_path=MODEL_SAVE_PATH):
    """
    Takes an unannotated CT scan image path, runs detection, and returns:
    - Bounding box coordinates (xmin, ymin, xmax, ymax) in original image pixels
    - Carcinoma classification name
    - Confidence probability percentage
    - Annotated image array with drawn bounding box and label text
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train first.")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = DualHeadLungCNN(num_classes=len(checkpoint["class_names"])).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Cannot read image at {image_path}")
    h, w, _ = orig_img.shape

    gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        pred_boxes, pred_logits = model(tensor)
        probs = torch.softmax(pred_logits, dim=1).cpu().numpy()[0]
        box = pred_boxes.cpu().numpy()[0]

    cls_idx = int(np.argmax(probs))
    conf = float(probs[cls_idx]) * 100.0
    cls_name = checkpoint["class_names"][cls_idx]

    # Convert normalized box back to original image pixel coordinates
    xmin = int(box[0] * w)
    ymin = int(box[1] * h)
    xmax = int(box[2] * w)
    ymax = int(box[3] * h)

    # Draw box & label on copy of original image
    annotated_img = orig_img.copy()
    cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    label_text = f"{cls_name}: {conf:.1f}%"
    cv2.putText(annotated_img, label_text, (xmin, max(20, ymin - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    return {
        "class_name": cls_name,
        "confidence_percent": conf,
        "box": (xmin, ymin, xmax, ymax),
        "annotated_image": annotated_img
    }

if __name__ == "__main__":
    train_model()
