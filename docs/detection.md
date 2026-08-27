# Detección de alimentos (FASE 2)

## Arquitectura

```
Imagen (multipart) → validación (MIME/tamaño/firma) → PIL → IFoodDetector.detect()
                                                              │
                                              YoloFoodDetector (ultralytics)
                                                              │
                                              DetectedFood[] (name, confidence, bbox px)
                                                              │
                                              AnalyzeResponse (snake_case)
```

- `IFoodDetector` (app/models/base.py): abstracción. `analyze.py` y los tests solo conocen la interfaz → cambiar YOLO por otro detector = nueva implementación.
- `YoloFoodDetector` (app/models/yolo_food_detector.py): wrapper de ultralytics, filtrado a clases de comida COCO.
- **Ciclo de vida**: el modelo se carga UNA vez en el lifespan de FastAPI (startup), no por request. `app.state.detector`.
- **Coordenadas**: bounding box en **píxeles** (x/y/width/height) de la imagen original — directo para dibujar en el frontend. No normalizadas.

## Endpoint

`POST /analyze` (multipart: `image`, `analysis_id`) → 200:

```json
{
  "analysis_id": "uuid",
  "status": "completed",
  "model_version": "food-detector-v1",
  "inference_time_ms": 182,
  "foods": [
    { "name": "pizza", "confidence": 0.94, "bounding_box": { "x": 120, "y": 80, "width": 300, "height": 180 } }
  ]
}
```

Errores: 400 `{detail: {success: false, error: {code, message}}}` (EMPTY_FILE, IMAGE_TOO_LARGE, INVALID_IMAGE, CORRUPT_FILE, INVALID_ANALYSIS_ID) · 503 MODEL_NOT_READY (modelo sin cargar).

## Configuración

| Variable | Default | Notas |
|---|---|---|
| `FOOD_AI_MODEL_PATH` | `weights/yolo11n.pt` | ultralytics descarga si no existe |
| `FOOD_AI_MODEL_VERSION` | `food-detector-v1` | se reporta en cada análisis |
| `FOOD_AI_CONFIDENCE_THRESHOLD` | `0.35` | experimentar 0.25–0.60 |
| `FOOD_AI_IMAGE_SIZE` | `640` | imgsz de inferencia |
| `FOOD_AI_DEVICE` | `cpu` | `cpu`/`cuda:0` |
| `FOOD_AI_MAX_DETECTIONS` | `20` | máx cajas por imagen |
| `FOOD_AI_DEBUG_IMAGES_DIR` | (vacío) | dir para overlays de debug |

## Debug visual

`app/utils/debug.py`: `draw_detections()` + `save_debug_image()` — imagen original + bounding boxes + labels + confidence. Activo con `FOOD_AI_DEBUG_IMAGES_DIR`; guarda `{analysis_id}.jpg`. Herramienta de desarrollo, NO parte de la respuesta API.

## Clases soportadas (COCO, honestas)

banana · apple · sandwich · orange · broccoli · carrot · hot dog · pizza · donut · cake

No se transforman nombres (pizza ≠ arroz): el resultado refleja las capacidades reales del modelo.

## Métricas durante desarrollo

Cada respuesta incluye `model_version` + `inference_time_ms`; el número de detecciones y confianzas se evalúan contra el threshold. Observabilidad completa en FASE 20.