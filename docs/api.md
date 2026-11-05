# API — Food AI

## Motor LLM de visión (opcional, bilingüe)

`FOOD_AI_PROVIDER=legacy|llm` selecciona el motor (default `legacy`). El motor
LLM reconoce alimentos de cualquier cocina mundial y estima cinco nutrientes
por 100 g. Estas aproximaciones solo se usan si USDA no tiene entrada.
`FOOD_AI_LLM_PROVIDER=anthropic|openai` (default `anthropic`) usa respectivamente
`claude-sonnet-4-6` o `gpt-4o`. `FOOD_AI_LLM_MODEL` configura el principal y
`FOOD_AI_LLM_MODEL_OPENAI` el backup. La cadena por defecto es Anthropic
(`FOOD_AI_LLM_API_KEY`) → OpenAI (si existe `FOOD_AI_LLM_API_KEY_OPENAI`) → legacy.
Las claves llegan exclusivamente por el entorno. `FOOD_AI_LLM_PROVIDER=openai`
permite elegir OpenAI directamente (acepta la clave específica o la principal).

`POST /analyze` conserva su multipart y añade campos opcionales:

```json
{
  "analysis_id": "uuid",
  "provider": "llm",
  "status": "completed",
  "foods": [{
    "name": "Sushi de salmón",
    "name_en": "Salmon sushi",
    "nutrition_query": "salmon sushi",
    "estimated_nutrients_per_100g": {
      "calories": 150, "protein_g": 10, "carbs_g": 20, "fat_g": 3, "fiber_g": 1
    },
    "confidence": 0.86,
    "bounding_box": {"x": 0, "y": 0, "width": 0, "height": 0},
    "segmentation": null,
    "portion": {
      "portion_size": "unknown",
      "estimated_grams": 200,
      "min_grams": 150,
      "max_grams": 250,
      "confidence": 0.7,
      "method": "llm_estimate"
    }
  }]
}
```

El ejemplo omite los campos de versión/tiempo existentes, que se conservan.
La caja vacía significa **sin localización** (no es una medición); no hay máscara.
El prompt usa `name_es` y `name_en`; el adaptador publica `name_es` como `name`.
El backend expone `nameEn`, `nutritionQuery` y `provider`; consulta USDA con
`nutritionQuery ?? name`. USDA siempre tiene prioridad (`nutritionSource: "usda"`).
Sin coincidencia, escala las estimaciones por los gramos y devuelve
`nutritionSource: "llm_estimate"`, `source: "llm_estimate"`, estado `available`.
Totales, rangos e intake suman ambas fuentes. Los cinco valores principales
siempre están presentes en un resultado LLM válido. Azúcar/sodio no se solicitan
al LLM y quedan desconocidos (`null`), incluso en totales mixtos; no se inventan
ceros. El origen se conserva en el snapshot existente sin migración de BD.
Legacy sin datos mantiene `unavailable` y edición manual con campos vacíos,
placeholder `—`; no existe modo demo ni nutrientes sintéticos de relleno.
La app usa `nameEn` en inglés cuando existe y conserva `name` como fallback.

Imagen JPEG en memoria, lado mayor ≤1568 px, sin persistir ni registrar imagen o
clave. Timeout total por proveedor de 30 s (`FOOD_AI_LLM_TIMEOUT_SECONDS`, máximo 30);
un único reintento con backoff de 0.5 s ante 429/5xx (`FOOD_AI_LLM_RETRIES=1`).
Timeout, JSON inválido, rechazo o credencial ausente producen WARN y fallback
al siguiente nivel; `provider` refleja el motor que produjo la respuesta.
Si el legacy tampoco está disponible, se conserva el error explícito 503 actual.
Una imagen sin comida devuelve `foods: []` con `provider: "llm"`, sin fallback.

`GET /health/readiness` y `GET /metrics` incluyen el proveedor configurado.
Métricas en memoria: `llm_calls` (análisis intentados, incluye credencial ausente)
y `llm_fallbacks` (fallos derivados a legacy). Se reinician con el proceso.
`llm_circuit_breakers` detalla por proveedor `state` (`closed|open|half_open`),
fallos consecutivos, segundos restantes, llamadas/fallos y saltos. Tras
`FOOD_AI_LLM_CIRCUIT_FAILURE_THRESHOLD=3` fallos se omite ese proveedor durante
`FOOD_AI_LLM_CIRCUIT_OPEN_SECONDS=300` segundos. Al expirar se permite una única
prueba: éxito cierra y fallo reabre el circuito. Cada worker mantiene su estado.
Readiness sigue comprobando los modelos locales necesarios para el fallback;
no valida la credencial remota.

Para probar localmente, inyectar la clave real fuera del historial de comandos,
exportar `FOOD_AI_PROVIDER=llm` y ejecutar `.venv\Scripts\python run_dev.py`
(8010, Windows). Enviar `image` + `analysis_id` UUID a `/analyze` o usar el backend.
Rollback: `FOOD_AI_PROVIDER=legacy` y reiniciar el proceso/tarea para recargar
el entorno; no necesita cambiar código ni reconstruir la imagen.
Backend/app tienen un presupuesto default de 120 s para ambos LLM + fallback.
Si una configuración existente sobrescribe el backend con un límite menor,
ajustar `FoodAi__TimeoutSeconds=120`. La prueba real 4.1 valida además gateway y
device. Un fallo total conserva el error explícito y el acceso al registro manual;
no se garantiza análisis automático si todos los motores están indisponibles.

Contratos consultados: [Messages de Anthropic](https://platform.claude.com/docs/en/api/http/messages)
y [Chat Completions de OpenAI](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

## Estado FASE 9 — Food Analysis / User Feedback

| Método | Ruta                                                      | Descripción                                                                          | Auth                                 |
| ------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------ |
| GET    | `/api/v1/foodai/health` (backend)                         | Probe backend → FoodAI Service                                                       | pública                              |
| POST   | `/api/v1/foodai/analyze` (backend)                        | Ingesta + pipeline completo + nutrición + persistencia del análisis                  | pública (persiste userId si hay JWT) |
| GET    | `/api/v1/foodai/nutrition/{foodKey}` (backend)            | Nutrición por 100 g                                                                  | pública                              |
| GET    | `/api/v1/foodai/analyses/{analysisId}` (backend)          | Recupera análisis persistido (items + snapshot + feedbacks)                          | [Authorize] + ownership              |
| POST   | `/api/v1/foodai/analyses/{analysisId}/feedback` (backend) | Corrección del usuario (FOOD_WRONG/PORTION_WRONG/DETECTION_WRONG/MISSING_FOOD/OTHER) | [Authorize] + ownership              |

| Método | Ruta                                           | Descripción                                                                                                                                                                | Auth                 |
| ------ | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| GET    | `/api/v1/foodai/health` (backend)              | Probe backend → FoodAI Service: `{backend, foodAI, detail}`                                                                                                                | pública              |
| POST   | `/api/v1/foodai/analyze` (backend)             | Ingesta + detección + segmentación + clasificación: multipart `image` → `{analysisId, status, modelVersion, segModelVersion, classifierVersion, inferenceTimeMs, foods[]}` | pública              |
| GET    | `/api/v1/foodai/nutrition/{foodKey}` (backend) | Nutrición por 100 g (alias YOLO o nombre canónico) — BD USDA FDC, sin IA                                                                                                   | pública              |
| GET    | `/health` (food-ai-service, puerto 8010)       | Salud del servicio + estado del modelo                                                                                                                                     | pública              |
| POST   | `/analyze` (food-ai-service, puerto 8010)      | Contrato interno backend → Python (multipart `image` + `analysis_id`)                                                                                                      | canal interno futuro |

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
      "portion": {
        "portionSize": "large",
        "estimatedGrams": 128,
        "minGrams": 118,
        "maxGrams": 160,
        "confidence": 0.55,
        "method": "basic_reference"
      },
      "nutrition": {
        "calories": 340.48,
        "protein": 14.58,
        "carbohydrates": 42.66,
        "fat": 13.31,
        "fiber": 2.94,
        "sugar": 4.61,
        "sodium": 765.44
      },
      "nutritionRange": {
        "min": { "calories": 313.88 },
        "max": { "calories": 425.6 }
      },
      "nutritionStatus": "available",
      "source": "USDA FoodData Central",
      "sourceVersion": "2026-08-27"
    }
  ],
  "summary": {
    "calories": 340.48,
    "protein": 14.58,
    "carbohydrates": 42.66,
    "fat": 13.31,
    "fiber": 2.94,
    "sugar": 4.61,
    "sodium": 765.44
  },
  "summaryRange": {
    "min": { "calories": 313.88 },
    "max": { "calories": 425.6 }
  },
  "intake": {
    "calories": 340.48,
    "proteinG": 14.58,
    "carbsG": 42.66,
    "fatG": 13.31,
    "fiberG": 2.94
  }
}
```

`intake` = totales listos para registrar como ingesta (`calories/proteinG/carbsG/fatG/fiberG`); `null` cuando ningún alimento tiene nutrición disponible (el cliente distingue "sin datos" de "ceros"). `GET /api/v1/foodai/analyses/{id}` devuelve el mismo objeto.

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
      "nutrition": {
        "calories": 234,
        "carbohydrates": 50.4,
        "protein": 4.86,
        "fat": 0.54
      }
    }
  ],
  "totals": {
    "calories": 234,
    "carbohydrates": 50.4,
    "protein": 4.86,
    "fat": 0.54
  }
}
```

### Errores

Respuestas consistentes, nunca errores internos sin procesar:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE",
    "message": "The uploaded image is not supported."
  }
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
