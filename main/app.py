"""
FastAPI Server & Interactive Web UI for Lung Tumor Detection and Subtyping
Final Year Project: Lung-PET-CT-Dx AI Diagnostic System
"""

import os
import io
import time
import json
import asyncio
import base64
import cv2
import numpy as np
import pydicom
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from train_detector import DualHeadLungCNN, IMG_SIZE, CLASS_NAMES
from dataset_pipeline import apply_lung_window

# --- Configuration ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "best_lung_tumor_model.pth")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = r"C:\Users\verma\Downloads\Lung Pet CT scan\best_lung_tumor_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sample_path(subpath, fallback):
    p = os.path.join(CURRENT_DIR, "sample_images", subpath)
    return p if os.path.exists(p) else fallback

SAMPLE_MAP = {
    "adenocarcinoma": get_sample_path("adenocarcinoma_sample.png", r"C:\Users\verma\Downloads\Lung Pet CT scan\dataset_png\images\A0002_1.3.6.1.4.1.14519.5.2.1.6655.2359.476982701137697438217958989680.png"),
    "squamous": get_sample_path("squamous_sample.png", r"C:\Users\verma\Downloads\Lung Pet CT scan\dataset_png\images\G0001_1.3.6.1.4.1.14519.5.2.1.6655.2359.198421860600941173627259058000.png"),
    "small_cell": get_sample_path("small_cell_sample.png", r"C:\Users\verma\Downloads\Lung Pet CT scan\dataset_png\images\B0001_1.3.6.1.4.1.14519.5.2.1.6655.2359.506419110499703332919918428911.png"),
    "large_cell": get_sample_path("large_cell_sample.png", r"C:\Users\verma\Downloads\Lung Pet CT scan\dataset_png\images\E0002_1.3.6.1.4.1.14519.5.2.1.6655.2359.142294656749345659159185600282.png")
}

# --- Load Model Checkpoint ---
print(f"[*] Loading model from {MODEL_PATH} on {DEVICE}...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = DualHeadLungCNN(num_classes=len(checkpoint["class_names"])).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("[+] Model loaded successfully!")

app = FastAPI(title="Lung Tumor AI Diagnostic System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def encode_img_base64(img_bgr):
    """Encodes BGR numpy image to base64 jpeg data URL."""
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        return ""
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def run_inference_on_array(orig_bgr):
    """Runs inference on a BGR image array and returns metadata & visualization."""
    t0 = time.time()
    h, w, _ = orig_bgr.shape

    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
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

    # Convert coordinates back to pixel space
    xmin = int(box[0] * w)
    ymin = int(box[1] * h)
    xmax = int(box[2] * w)
    ymax = int(box[3] * h)

    # Ensure box validity
    xmin = max(0, min(xmin, w - 1))
    ymin = max(0, min(ymin, h - 1))
    xmax = max(xmin + 1, min(xmax, w))
    ymax = max(ymin + 1, min(ymax, h))

    # Render bounding box and label
    annotated = orig_bgr.copy()
    
    # Glow / outline box
    cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    
    # Label badge with background pill
    label_text = f"{cls_name} ({conf:.1f}%)"
    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    badge_ymin = max(0, ymin - text_h - 10)
    cv2.rectangle(annotated, (xmin, badge_ymin), (xmin + text_w + 10, ymin), (0, 0, 220), -1)
    cv2.putText(annotated, label_text, (xmin + 5, ymin - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    elapsed_ms = round((time.time() - t0) * 1000, 1)

    prob_dict = {
        name: round(float(probs[i]) * 100.0, 2)
        for i, name in enumerate(checkpoint["class_names"])
    }

    return {
        "status": "success",
        "class_name": cls_name,
        "confidence_percent": round(conf, 2),
        "probabilities": prob_dict,
        "box": {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "width": xmax - xmin,
            "height": ymax - ymin
        },
        "latency_ms": elapsed_ms,
        "original_image": encode_img_base64(orig_bgr),
        "annotated_image": encode_img_base64(annotated)
    }

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Accepts uploaded PNG, JPG, or DICOM (.dcm) files with file-size safeguards."""
    # Safety constraint: enforce maximum 25 MB payload to protect memory
    MAX_FILE_SIZE = 25 * 1024 * 1024
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Payload too large. Maximum supported scan size is 25 MB."
        )

    filename = file.filename.lower()

    if filename.endswith(".dcm"):
        # Handle DICOM file
        try:
            dcm_io = io.BytesIO(contents)
            ds = pydicom.dcmread(dcm_io)
            slope = getattr(ds, "RescaleSlope", 1)
            intercept = getattr(ds, "RescaleIntercept", 0)
            norm_gray = apply_lung_window(ds.pixel_array, slope, intercept)
            orig_bgr = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse DICOM: {str(e)}")
    else:
        # Standard image (PNG / JPG)
        nparr = np.frombuffer(contents, np.uint8)
        orig_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if orig_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

    return run_inference_on_array(orig_bgr)

@app.post("/predict/stream")
async def predict_stream_endpoint(file: UploadFile = File(...)):
    """
    Streaming FastAPI endpoint (Server-Sent Events) providing real-time diagnostic
    telemetry and progressive inference updates as the scan moves through the CAD pipeline.
    """
    MAX_FILE_SIZE = 25 * 1024 * 1024
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Payload too large. Maximum supported scan size is 25 MB."
        )

    filename = file.filename.lower()

    async def event_generator():
        yield f"data: {json.dumps({'stage': 'upload', 'message': 'Scan received and validated successfully.', 'progress': 20})}\n\n"
        await asyncio.sleep(0.01)

        try:
            if filename.endswith(".dcm"):
                yield f"data: {json.dumps({'stage': 'dicom_parsing', 'message': 'Extracting DICOM metadata & applying Hounsfield lung windowing...', 'progress': 40})}\n\n"
                await asyncio.sleep(0.01)
                dcm_io = io.BytesIO(contents)
                ds = pydicom.dcmread(dcm_io)
                slope = getattr(ds, "RescaleSlope", 1)
                intercept = getattr(ds, "RescaleIntercept", 0)
                norm_gray = apply_lung_window(ds.pixel_array, slope, intercept)
                orig_bgr = cv2.cvtColor(norm_gray, cv2.COLOR_GRAY2BGR)
            else:
                yield f"data: {json.dumps({'stage': 'preprocessing', 'message': 'Decoding standard scan & normalizing tensor dimensions...', 'progress': 40})}\n\n"
                await asyncio.sleep(0.01)
                nparr = np.frombuffer(contents, np.uint8)
                orig_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if orig_bgr is None:
                    yield f"data: {json.dumps({'error': 'Invalid image format.'})}\n\n"
                    return

            yield f"data: {json.dumps({'stage': 'inference', 'message': 'Executing Dual-Head CNN (Localization + Subtyping)...', 'progress': 75})}\n\n"
            await asyncio.sleep(0.01)

            # Perform PyTorch inference
            result = run_inference_on_array(orig_bgr)

            yield f"data: {json.dumps({'stage': 'complete', 'message': 'Inference complete.', 'progress': 100, 'result': result})}\n\n"
        except Exception as ex:
            yield f"data: {json.dumps({'error': str(ex)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/sample/{sample_key}")
async def get_sample(sample_key: str):
    """Loads a pre-stored sample slice from the real dataset."""
    sample_key = sample_key.lower()
    if sample_key not in SAMPLE_MAP:
        raise HTTPException(status_code=404, detail=f"Sample '{sample_key}' not found.")

    img_path = SAMPLE_MAP[sample_key]
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Sample image file missing on disk.")

    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None:
        try:
            orig_bgr = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            pass

    if orig_bgr is None:
        raise HTTPException(status_code=500, detail="Unable to decode sample scan image.")

    return run_inference_on_array(orig_bgr)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the complete interactive diagnostic UI."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lung Cancer AI Detection System | Final Year Project</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-card: rgba(18, 25, 41, 0.75);
            --border-card: rgba(255, 255, 255, 0.08);
            --accent-cyan: #00e5ff;
            --accent-red: #ff3366;
            --accent-green: #00e676;
            --text-main: #f0f4f8;
            --text-dim: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at 50% 0%, #152238 0%, #0a0e17 70%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 24px;
        }

        .container {
            max-width: 1280px;
            margin: 0 auto;
        }

        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-card);
            margin-bottom: 28px;
        }

        .logo-area {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .logo-icon {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, var(--accent-cyan), #0077ff);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            box-shadow: 0 0 20px rgba(0, 229, 255, 0.35);
        }

        .title-block h1 {
            font-size: 1.35rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }

        .title-block p {
            font-size: 0.85rem;
            color: var(--text-dim);
        }

        .badge-status {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(0, 230, 118, 0.12);
            color: var(--accent-green);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            border: 1px solid rgba(0, 230, 118, 0.25);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-green);
            box-shadow: 0 0 10px var(--accent-green);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.4; transform: scale(0.85); }
        }

        .grid {
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 24px;
        }

        @media (max-width: 960px) {
            .grid { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 16px;
            padding: 24px;
            backdrop-filter: blur(16px);
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.35);
        }

        .card h2 {
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Upload Area */
        .dropzone {
            border: 2px dashed rgba(0, 229, 255, 0.3);
            border-radius: 14px;
            padding: 36px 20px;
            text-align: center;
            background: rgba(0, 229, 255, 0.02);
            cursor: pointer;
            transition: all 0.25s ease;
            position: relative;
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--accent-cyan);
            background: rgba(0, 229, 255, 0.08);
            transform: translateY(-2px);
        }

        .dropzone-icon {
            font-size: 40px;
            margin-bottom: 10px;
        }

        .dropzone p {
            font-size: 0.9rem;
            color: var(--text-dim);
        }

        .dropzone span {
            color: var(--accent-cyan);
            font-weight: 600;
        }

        .dropzone input[type="file"] {
            position: absolute;
            inset: 0;
            opacity: 0;
            cursor: pointer;
        }

        /* Quick Sample Buttons */
        .sample-section {
            margin-top: 24px;
        }

        .sample-section h3 {
            font-size: 0.85rem;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }

        .sample-buttons {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }

        .btn-sample {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-card);
            color: var(--text-main);
            padding: 10px 12px;
            border-radius: 8px;
            font-size: 0.8rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
        }

        .btn-sample:hover {
            background: rgba(0, 229, 255, 0.12);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }

        /* Results Display */
        .results-container {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .image-viewer {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        @media (max-width: 680px) {
            .image-viewer { grid-template-columns: 1fr; }
        }

        .image-box {
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-card);
            position: relative;
            display: flex;
            flex-direction: column;
        }

        .image-box-header {
            padding: 8px 14px;
            background: rgba(255, 255, 255, 0.05);
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            color: var(--text-dim);
            display: flex;
            justify-content: space-between;
        }

        .image-box img {
            width: 100%;
            height: auto;
            aspect-ratio: 1/1;
            object-fit: contain;
            display: block;
        }

        .placeholder-box {
            aspect-ratio: 1/1;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-dim);
            font-size: 0.9rem;
            background: rgba(0, 0, 0, 0.4);
        }

        /* Diagnosis Panel */
        .diagnosis-card {
            background: linear-gradient(135deg, rgba(255, 51, 102, 0.08) 0%, rgba(18, 25, 41, 0.9) 100%);
            border: 1px solid rgba(255, 51, 102, 0.25);
            border-radius: 14px;
            padding: 18px 22px;
            display: grid;
            grid-template-columns: auto 1fr auto;
            gap: 20px;
            align-items: center;
        }

        .diagnosis-pill {
            background: var(--accent-red);
            color: #fff;
            padding: 8px 14px;
            border-radius: 8px;
            font-weight: 700;
            font-size: 1.1rem;
            letter-spacing: -0.3px;
            box-shadow: 0 4px 14px rgba(255, 51, 102, 0.4);
        }

        .diagnosis-meta h4 {
            font-size: 0.8rem;
            color: var(--text-dim);
            text-transform: uppercase;
            margin-bottom: 4px;
        }

        .diagnosis-meta .tumor-type {
            font-size: 1.25rem;
            font-weight: 700;
            color: #fff;
        }

        .latency-badge {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: var(--text-dim);
            background: rgba(255, 255, 255, 0.05);
            padding: 4px 10px;
            border-radius: 6px;
        }

        /* Probability Bars */
        .prob-section {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .prob-row {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .prob-label-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
        }

        .prob-bar-bg {
            height: 8px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 4px;
            overflow: hidden;
        }

        .prob-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #0077ff, var(--accent-cyan));
            border-radius: 4px;
            transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .prob-bar-fill.highlight {
            background: linear-gradient(90deg, #ff0055, var(--accent-red));
        }

        /* Loading Spinner */
        .spinner {
            display: none;
            width: 32px;
            height: 32px;
            border: 3px solid rgba(0, 229, 255, 0.2);
            border-top-color: var(--accent-cyan);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 20px auto;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-area">
                <div class="logo-icon">🫁</div>
                <div class="title-block">
                    <h1>Lung-PET-CT-Dx AI Diagnostic System</h1>
                    <p>Final Year Project: Automated Tumor Localization & Histopathological Subtyping</p>
                </div>
            </div>
            <div class="badge-status">
                <div class="status-dot"></div>
                <span>Dual-Head CNN Online (92.5% Val Acc)</span>
            </div>
        </header>

        <div class="grid">
            <!-- Left Panel: Controls & Upload -->
            <div class="card">
                <h2><span>📤</span> Upload CT Scan Slice</h2>
                <div class="dropzone" id="dropzone">
                    <div class="dropzone-icon">📁</div>
                    <p>Drag & drop unannotated scan here or <span>browse</span></p>
                    <p style="font-size: 0.75rem; margin-top: 6px; opacity: 0.7;">Supports .png, .jpg, or raw .dcm (DICOM)</p>
                    <input type="file" id="fileInput" accept=".png,.jpg,.jpeg,.dcm">
                </div>

                <div class="spinner" id="spinner"></div>

                <div class="sample-section">
                    <h3>⚡ Or Test Real Dataset Samples</h3>
                    <div class="sample-buttons">
                        <button class="btn-sample" onclick="loadSample('adenocarcinoma')">🔬 Adeno (A)</button>
                        <button class="btn-sample" onclick="loadSample('squamous')">🧬 Squamous (G)</button>
                        <button class="btn-sample" onclick="loadSample('small_cell')">🧪 Small Cell (B)</button>
                        <button class="btn-sample" onclick="loadSample('large_cell')">🧫 Large Cell (E)</button>
                    </div>
                </div>

                <div style="margin-top: 24px; padding-top: 18px; border-top: 1px solid var(--border-card); font-size: 0.78rem; color: var(--text-dim); line-height: 1.5;">
                    💡 <b>How it works:</b> The dual-head CNN performs simultaneous regression (locating the bounding box coordinates) and multi-class histopathology subtyping.
                </div>
            </div>

            <!-- Right Panel: Results & Visualization -->
            <div class="card results-container">
                <h2><span>🔍</span> Real-Time AI Detection & Subtyping</h2>

                <!-- Side-by-side Images -->
                <div class="image-viewer">
                    <div class="image-box">
                        <div class="image-box-header">
                            <span>Input Unannotated Slice</span>
                            <span id="origDims">--</span>
                        </div>
                        <div id="origPlaceholder" class="placeholder-box">Upload or select a sample scan</div>
                        <img id="origImg" style="display:none;" alt="Original Scan">
                    </div>

                    <div class="image-box">
                        <div class="image-box-header">
                            <span style="color: var(--accent-cyan);">AI Annotated Tumor Localization</span>
                            <span id="boxCoords" style="font-family: 'JetBrains Mono', monospace;">--</span>
                        </div>
                        <div id="annotPlaceholder" class="placeholder-box">Awaiting scan input</div>
                        <img id="annotImg" style="display:none;" alt="Annotated Prediction">
                    </div>
                </div>

                <!-- Diagnosis Card -->
                <div class="diagnosis-card" id="diagCard" style="display: none;">
                    <div class="diagnosis-pill" id="diagPercent">0.0%</div>
                    <div class="diagnosis-meta">
                        <h4>Diagnostic Output</h4>
                        <div class="tumor-type" id="diagType">--</div>
                    </div>
                    <div class="latency-badge" id="diagLatency">-- ms</div>
                </div>

                <!-- Probability Distribution -->
                <div class="card" id="probCard" style="display: none; padding: 18px; background: rgba(0,0,0,0.25);">
                    <h3 style="font-size: 0.85rem; margin-bottom: 12px; color: var(--text-dim); text-transform: uppercase;">Confidence Probability Distribution</h3>
                    <div class="prob-section" id="probContainer"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const dropzone = document.getElementById('dropzone');
        const spinner = document.getElementById('spinner');

        const origImg = document.getElementById('origImg');
        const annotImg = document.getElementById('annotImg');
        const origPlaceholder = document.getElementById('origPlaceholder');
        const annotPlaceholder = document.getElementById('annotPlaceholder');

        const diagCard = document.getElementById('diagCard');
        const diagType = document.getElementById('diagType');
        const diagPercent = document.getElementById('diagPercent');
        const diagLatency = document.getElementById('diagLatency');
        const boxCoords = document.getElementById('boxCoords');

        const probCard = document.getElementById('probCard');
        const probContainer = document.getElementById('probContainer');

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                uploadFile(e.target.files[0]);
            }
        });

        async function uploadFile(file) {
            const formData = new FormData();
            formData.append('file', file);

            setLoading(true);
            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                if (!res.ok) throw new Error(await res.text());
                const data = await res.json();
                renderResult(data);
            } catch (err) {
                alert('Inference Error: ' + err.message);
            } finally {
                setLoading(false);
            }
        }

        async function loadSample(sampleKey) {
            setLoading(true);
            try {
                const res = await fetch('/api/sample/' + sampleKey);
                if (!res.ok) throw new Error(await res.text());
                const data = await res.json();
                renderResult(data);
            } catch (err) {
                alert('Sample Error: ' + err.message);
            } finally {
                setLoading(false);
            }
        }

        function setLoading(isLoading) {
            spinner.style.display = isLoading ? 'block' : 'none';
        }

        function renderResult(data) {
            origPlaceholder.style.display = 'none';
            annotPlaceholder.style.display = 'none';
            origImg.style.display = 'block';
            annotImg.style.display = 'block';

            origImg.src = data.original_image;
            annotImg.src = data.annotated_image;

            diagCard.style.display = 'grid';
            diagType.innerText = data.class_name;
            diagPercent.innerText = data.confidence_percent.toFixed(1) + '%';
            diagLatency.innerText = data.latency_ms + ' ms';

            boxCoords.innerText = `Box: [${data.box.xmin}, ${data.box.ymin}, ${data.box.xmax}, ${data.box.ymax}]`;

            // Probability bars
            probCard.style.display = 'block';
            probContainer.innerHTML = '';
            for (const [cname, prob] of Object.entries(data.probabilities)) {
                const isWinner = (cname === data.class_name);
                const row = document.createElement('div');
                row.className = 'prob-row';
                row.innerHTML = `
                    <div class="prob-label-row">
                        <span style="${isWinner ? 'font-weight:700; color:var(--text-main);' : 'color:var(--text-dim);'}">${cname}</span>
                        <span style="font-family:'JetBrains Mono', monospace; font-weight:600;">${prob.toFixed(1)}%</span>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill ${isWinner ? 'highlight' : ''}" style="width: ${prob}%;"></div>
                    </div>
                `;
                probContainer.appendChild(row);
            }
        }
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
