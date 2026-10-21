# Stage 1: Base image with lightweight Python
FROM python:3.10-slim

# Prevent interactive prompts and Python byte-compile clutter
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install native system libraries required by OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-optimized PyTorch and Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, trained model weights, and sample data
COPY app.py .
COPY train_detector.py .
COPY dataset_pipeline.py .
COPY best_lung_tumor_model.pth .
COPY sample_images/ ./sample_images/

# Expose FastAPI port
EXPOSE 8000

# Health check to ensure the container is running and responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Start FastAPI application via Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
