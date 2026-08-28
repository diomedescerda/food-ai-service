# API — Food AI

## Estado FASE 9 — Food Analysis / User Feedback

| Método | Ruta | Descripción | Auth |
|---|---|---|---|
| GET | `/api/v1/foodai/health` (backend) | Probe backend → FoodAI Service | pública |
| POST | `/api/v1/foodai/analyze` (backend) | Ingesta + pipeline completo + nutrición + persistencia del análisis | pública (persiste userId si hay JWT) |
| GET | `/api/v1/foodai/nutrition/{foodKey}` (backend) | Nutrición por 100 g | pública |
| GET | `/api/v1/foodai/analyses/{analysisId}` (backend) | Recupera análisis persistido (items + snapshot + feedbacks) | [Authorize] + ownership |
| POST | `/api/v1/foodai/analyses/{analysisId}/feedback` (backend) | Corrección del usuario (FOOD_WRONG/PORTION_WRONG/DETECTION_WRONG/MISSING_FOOD/OTHER) | [Authorize] + ownership |

| Método | Ruta | Descripción | Auth |
|---|---|---|---|
| GET | `/api/v1/foodai/health` (backend) | Probe backend → FoodAI Service: `{backend, foodAI, detail}` | pública |
| POST | `/api/v1/foodai/analyze` (backend) | Ingesta + detección + segmentación + clasificación: multipart `image` → `{analysisId, status, modelVersion, segModelVersion, classifierVersion, inferenceTimeMs, foods[]}` | pública |
| GET | `/api/v1/foodai/nutrition/{foodKey}` (backend) | Nutrición por 100 g (alias YOLO o nombre canónico) — BD USDA FDC, sin IA | pública |
| GET | `/health` (food-ai-service, puerto 8010) | Salud del servicio + estado del modelo | pública |
| POST | `/analyze` (food-ai-service, puerto 8010) | Contrato interno backend → Python (multipart `image` + `analysis_id`) | canal interno futuro |

### Respuestas

`POST /api/v1/foodai/analyze` (multipart `image=<archivo>`):
```json
{
  "analysisId": "d7637c16-80a0-4947-b63e-833de226012b",
  "status": "completed",
  "modelVersion": "food-detector-v1",
  "segModelVersion": "food-segmenter-v1",
  "classifierVersion": "detector-based-v1",
  "inferenceTimeMs": 5185,
  "foods": [
    {
      "name": "pizza",
      "confidence": 0.9253,
      "boundingBox": { "x": 7, "y": 14, "width": 318, "height": 216 },
      "segmentation": { "mask": "<base64 PNG>", "areaPixels": 44970 },
      "portion": { "portionSize": "large", "estimatedGrams": 128, "minGrams": 118, "maxGrams": 160, "confidence": 0.55, "method": "basic_reference" },
      "nutrition": { "calories": 340.48, "protein": 14.58, "carbohydrates": 42.66, "fat": 13.31, "fiber": 2.94, "sugar": 4.61, "sodium": 765.44 },
      "nutritionRange": { "min": { "calories": 313.88 }, "max": { "calories": 425.60 } },
      "nutritionStatus": "available",
      "source": "USDA FoodData Central",
      "sourceVersion": "2026-08-27"
    }
  ],
  "summary": { "calories": 340.48, "protein": 14.58, "carbohydrates": 42.66, "fat": 13.31, "fiber": 2.94, "sugar": 4.61, "sodium": 765.44 },
  "summaryRange": { "min": { "calories": 313.88 }, "max": { "calories": 425.60 } }
}
```

⚠️ `estimatedGrams` y la nutrición derivada son ESTIMACIONES, no mediciones.

`nutritionStatus`: `available` | `unavailable` (sin entrada en la DB, ej. sandwich) | `portion_unavailable` (sin gramos). Alimentos sin nutrición no rompen el análisis y quedan visibles; los totales solo suman los disponibles.

## Persistencia y feedback

`POST /analyze` persiste el análisis (schema `foodai.`): items con snapshot nutricional + versiones de modelo + máscara en object storage (`maskKey`), nunca base64 en PG. Idempotente por analysisId.

`POST /api/v1/foodai/analyses/{id}/feedback`:
```json
{
  "itemIndex": 0,
  "type": "PORTION_WRONG",
  "correctedGrams": 200,
  "note": "era más"
}
```
→ `201` con `originalGrams: 128, correctedGrams: 200` (el original nunca se sobrescribe). `401` sin token; `404` para análisis ajenos (ownership).

`GET /health` (food-ai-service):
```json
{
  "status": "healthy",
  "service": "food-ai-service",
  "version": "0.1.0",
  "timestamp_utc": "...",
  "model": { "loaded": true, "version": "food-detector-v1" },
  "segmentation_model": { "loaded": true, "version": "food-segmenter-v1" },
  "classifier_model": { "loaded": true, "version": "detector-based-v1" }
}
```

Errores backend (400): `{"error":{"code":"EMPTY_FILE"|"IMAGE_TOO_LARGE"|"INVALID_IMAGE"|"CORRUPT_FILE","message":"..."}}`
502: `{"error":{"code":"AI_SERVICE_UNAVAILABLE","message":"..."}}`

### Validación de imágenes (backend, `ImageFileValidator`)

- Extensión: `.jpg .jpeg .png .webp`
- MIME: `image/jpeg image/png image/webp` (parámetros normalizados)
- Tamaño máx: `FoodAi:MaxImageSizeBytes` (default 10 MB)
- Vacío: longitud 0
- Corrupto: firma mágica (JPEG `FF D8 FF`, PNG `89 50 4E 47`, WEBP `RIFF....WEBP`)

### Storage

`IImageStorage` → `LocalImageStorage` (delega en `IObjectStorageService` existente: Local o S3 vía `Storage:Provider`). Clave canónica `foodai/<analysisId>.<ext>` → `CoppAddresd/.local-storage/foodai/` en dev.

### Contrato .NET ↔ Python (versión 1.0)

El backend es el único que habla con el Food AI Service. Los schemas internos del servicio NO se exponen al frontend.

### Contrato de análisis (FASE 3, implementado)

Request (backend → Python, multipart):
```
image       → archivo de imagen
analysis_id → UUID
```

Response:
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

Errores Python (400): `{"detail": {"success": false, "error": {"code": "INVALID_ANALYSIS_ID"|"EMPTY_FILE"|"IMAGE_TOO_LARGE"|"INVALID_IMAGE"|"CORRUPT_FILE", "message": "..."}}}` · 503 `MODEL_NOT_READY`

### Contrato futuro (FASE 3+)

Response con detección:
```json
{
  "analysisId": "uuid",
  "status": "completed",
  "foods": [
    {
      "name": "rice",
      "confidence": 0.94,
      "estimatedGrams": 180,
      "portionConfidence": 0.68,
      "nutrition": { "calories": 234, "carbohydrates": 50.4, "protein": 4.86, "fat": 0.54 }
    }
  ],
  "totals": { "calories": 234, "carbohydrates": 50.4, "protein": 4.86, "fat": 0.54 }
}
```

### Errores

Respuestas consistentes, nunca errores internos sin procesar:
```json
{
  "success": false,
  "error": { "code": "INVALID_IMAGE", "message": "The uploaded image is not supported." }
}
```

### Endpoints futuros (FASE 16)

```
POST /api/food/analyze
GET  /api/food/analysis/{id}
POST /api/food/analysis/{id}/correction
GET  /api/foods
GET  /api/foods/{id}
GET  /api/nutrition/{foodId}
```