# Detección y segmentación de alimentos (FASE 2-3)

## Arquitectura

```
Imagen (multipart) → validación (MIME/tamaño/firma) → PIL → IFoodDetector.detect()
                                                              │
                                              YoloFoodDetector (ultralytics)
                                                              │
                                              DetectedFood[] (name, confidence, bbox px)
                                                              │
                                              IFoodSegmenter.segment() (por detección)
                                                              │
                                              YoloFoodSegmenter (yolo11n-seg)
                                                              │
                                              SegmentationResult (mask PNG b64 + area_pixels)
```

- `IFoodDetector` (app/models/base.py): abstracción de detección. `analyze.py` y los tests solo conocen la interfaz.
- `IFoodSegmenter` (app/models/segmenter_base.py): abstracción de segmentación, **separada del detector** — no mezclada en `YoloFoodDetector`.
- `YoloFoodSegmenter` (app/models/yolo_food_segmenter.py): wrapper de `yolo11n-seg.pt` (variante de segmentación de Ultralytics, misma familia que el detector). Empareja máscara↔detección por IoU del bbox (≥0.5).
- **Ciclo de vida**: detector y segmentador se cargan UNA vez en el lifespan de FastAPI (startup), no por request. `app.state.detector` / `app.state.segmenter`.
- **Coordenadas**: bounding box en **píxeles** (x/y/width/height) de la imagen original.

## Máscara de segmentación

- Formato: **PNG en escala de grises** (255 = alimento), recortado al bounding box, codificado en **base64** — ligera y directa para dibujar en el frontend sobre el bbox.
- `area_pixels`: píxeles de alimento dentro de la máscara (nunca el área del bbox).
- Máscaras verificadas en tests reales: dimensiones = bbox, área > 0, área < bbox completo.

## Endpoint

`POST /analyze` (multipart: `image`, `analysis_id`) → 200:

```json
{
  "analysis_id": "uuid",
  "status": "completed",
  "model_version": "food-detector-v1",
  "seg_model_version": "food-segmenter-v1",
  "inference_time_ms": 5185,
  "foods": [
    {
      "name": "pizza",
      "confidence": 0.9253,
      "bounding_box": { "x": 7, "y": 14, "width": 318, "height": 216 },
      "segmentation": { "mask": "<base64 PNG>", "area_pixels": 44970 }
    }
  ]
}
```

Errores: 400 (EMPTY_FILE, IMAGE_TOO_LARGE, INVALID_IMAGE, CORRUPT_FILE, INVALID_ANALYSIS_ID) · 503 MODEL_NOT_READY.

## Configuración

| Variable | Default | Notas |
|---|---|---|
| `FOOD_AI_MODEL_PATH` | `weights/yolo11n.pt` | detector |
| `FOOD_AI_MODEL_VERSION` | `food-detector-v1` | |
| `FOOD_AI_SEG_MODEL_PATH` | `weights/yolo11n-seg.pt` | segmentador |
| `FOOD_AI_SEG_MODEL_VERSION` | `food-segmenter-v1` | |
| `FOOD_AI_CONFIDENCE_THRESHOLD` | `0.35` | compartido |
| `FOOD_AI_IMAGE_SIZE` | `640` | imgsz compartido |
| `FOOD_AI_DEVICE` | `cpu` | |
| `FOOD_AI_MAX_DETECTIONS` | `20` | |
| `FOOD_AI_DEBUG_IMAGES_DIR` | (vacío) | overlays de debug |

## Debug visual

`app/utils/debug.py`: `draw_detections()` pinta máscara semi-transparente + bounding box + label + confidence; `save_debug_image()` guarda `{analysis_id}.jpg`. Herramienta de desarrollo, NO parte de la respuesta API.

## Clases soportadas (COCO, honestas)

banana · apple · sandwich · orange · broccoli · carrot · hot dog · pizza · donut · cake

## Métricas (línea base, CPU, 640px)

| Imagen | Detección+Seg total | Área máscara |
|---|---|---|
| pizza (330×247) | 5185 ms (frío) / ~200 ms (caliente) | 44970 px |
| banana (330×291) | 232 ms | 24629 px |
| apple (330×299) | ~266 ms | ~20000 px |