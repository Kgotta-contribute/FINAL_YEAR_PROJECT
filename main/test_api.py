"""
Automated unit tests for Lung Tumor AI Diagnostic API and streaming endpoints.
Used by GitHub Actions CI pipeline.
"""

import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from app import app, checkpoint

client = TestClient(app)

def test_home_ui():
    """Verify that clinical dashboard UI returns HTTP 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Lung Cancer AI Detection System" in response.text

def test_sample_endpoint():
    """Verify sample scan retrieval and inference."""
    response = client.get("/api/sample/adenocarcinoma")
    assert response.status_code == 200
    data = response.json()
    assert "class_name" in data
    assert "confidence_percent" in data
    assert "box" in data
    assert "latency_ms" in data
    assert "xmin" in data["box"]

def test_predict_endpoint_with_synthetic_scan():
    """Verify POST /predict with a synthetic CT scan slice."""
    # Create a 512x512 dummy scan
    dummy_img = np.ones((512, 512, 3), dtype=np.uint8) * 128
    _, encoded = cv2.imencode(".png", dummy_img)
    
    response = client.post(
        "/predict",
        files={"file": ("test_slice.png", encoded.tobytes(), "image/png")}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["class_name"] in checkpoint["class_names"]
    assert 0.0 <= payload["confidence_percent"] <= 100.0

def test_streaming_endpoint():
    """Verify POST /predict/stream returns real-time SSE progress events."""
    dummy_img = np.ones((256, 256, 3), dtype=np.uint8) * 100
    _, encoded = cv2.imencode(".png", dummy_img)
    
    response = client.post(
        "/predict/stream",
        files={"file": ("stream_test.png", encoded.tobytes(), "image/png")}
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    content = response.text
    assert "upload" in content
    assert "complete" in content
