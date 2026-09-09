FROM python:3.12-slim AS final
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

# Libs runtime de OpenCV (base slim no las trae).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libnss3 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Torch solo CPU: evita wheels CUDA de PyPI (imagen multi-GB).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Pesos: el repo NO incluye los .pt (.gitignore/.dockerignore). Se descargan
# los oficiales preentrenados de ultralytics en build (cache de capas).
# En prod se pueden MONTAR entrenados desde un volumen /app/weights.
RUN python -c "from ultralytics import YOLO; \
    YOLO('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'); \
    YOLO('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt')"

COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
