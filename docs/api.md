# API — Food AI

## Estado FASE 1 — Image Ingestion

| Método | Ruta | Descripción | Auth |
|---|---|---|---|
| GET | `/api/v1/foodai/health` (backend) | Probe backend → FoodAI Service: `{backend, foodAI, detail}` | pública |
| POST | `/api/v1/foodai/analyze` (backend) | Ingesta de imagen: multipart `image` → `{analysisId, status}` | pública |
| GET | `/health` (food-ai-service, puerto 8010) | Salud del servicio Python | pública |
| POST | `/analyze` (food-ai-service, puerto 8010) | Contrato interno backend → Python (multipart `image` + `analysis_id`) | canal interno futuro |

### Respuestas

`GET /api/v1/foodai/health`:
```json
{
  "backend": "healthy",
  "foodAI": "healthy",
  "detail": "food-ai-service"
}
```

`POST /api/v1/foodai/analyze` (multipart `image=<archivo>`):
```json
{
  "analysisId": "61b98e10-6215-4d02-9216-0da8728c943c",
  "status": "received"
}
```

Errores (400): `{"error":{"code":"EMPTY_FILE"|"IMAGE_TOO_LARGE"|"INVALID_IMAGE"|"CORRUPT_FILE","message":"..."}}`
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

### Contrato de análisis (FASE 1, implementado)

Request (backend → Python, multipart):
```
image       → archivo de imagen
analysis_id → UUID
```

Response:
```json
{
  "analysis_id": "uuid",
  "status": "received"
}
```

Errores Python (400): `{"detail": {"success": false, "error": {"code": "INVALID_ANALYSIS_ID"|"EMPTY_FILE"|"IMAGE_TOO_LARGE"|"INVALID_IMAGE"|"CORRUPT_FILE", "message": "..."}}}`

### Contrato futuro (FASE 2+)

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