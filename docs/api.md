# API — Food AI

## Estado FASE 0

| Método | Ruta | Descripción | Auth |
|---|---|---|---|
| GET | `/api/v1/foodai/health` (backend) | Probe backend → FoodAI Service: `{backend, foodAI, detail}` | pública |
| GET | `/health` (food-ai-service, puerto 8010) | Salud del servicio Python | pública |
| GET | `/health` (backend existente) | Health global (DB, etc.) | pública |

### Respuestas

`GET /api/v1/foodai/health`:
```json
{
  "backend": "healthy",
  "foodAI": "healthy",
  "detail": "food-ai-service"
}
```

`GET /health` (food-ai-service):
```json
{
  "status": "healthy",
  "service": "food-ai-service",
  "version": "0.1.0",
  "timestamp_utc": "2026-08-27T19:07:14.518189Z"
}
```

## Contrato .NET ↔ Python (versión 1.0)

El backend es el único que habla con el Food AI Service. Los schemas internos del servicio NO se exponen al frontend.

### Contrato de análisis (FASE 1+, borrador)

Request (backend → Python):
```json
{
  "analysisId": "uuid",
  "imageUrl": "..."
}
```

Response:
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