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

# Pre-bake de los modelos HuggingFace del runtime (CLIP zero-shot, Grounding
# DINO del detector híbrido y DINOv2 del pipeline F51). Sin esto, el primer
# arranque del task descarga ~1.8 GB en caliente y puede exceder el grace
# period de ECS (150 s) → task unhealthy en loop.
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"
RUN python -c "from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor; \
    AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny'); \
    AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')"
RUN python -c "from transformers import AutoModel, AutoImageProcessor; \
    AutoModel.from_pretrained('facebook/dinov2-base'); \
    AutoImageProcessor.from_pretrained('facebook/dinov2-base')"

COPY app/ app/
# Catálogo 5.761 alimentos + embeddings multi-text (pipeline F51) y mapping
# nutricional precomputado (NutritionService). Requeridos por
# FOOD_AI_RETRIEVAL_ENABLED/FOOD_AI_NUTRITION_ENABLED en producción.
COPY catalog/ catalog/
COPY nutrition/ nutrition/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
