# FoodAI Service

Servicio de IA de análisis de alimentos del ecosistema CoppAddresd: detección, segmentación, clasificación, estimación de porción y nutrición. Python 3.12+ / FastAPI.

**Estado actual: FASE 8 — Nutrition Engine.** `POST /analyze` (vía backend) detecta, segmenta, clasifica, estima porción y calcula nutrición por alimento + totales (Nutrition DB en PostgreSQL, única fuente; cálculo decimal en .NET). Detalle: `docs/nutrition.md`, `docs/portion-estimation.md`.

## Comandos

```bash
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest        # tests (incluye YOLO real si hay modelo)
.\.venv\Scripts\python run_dev.py       # dev server puerto 8010 (Windows: selector loop)
```

**Modelos** (primera vez):
```bash
mkdir weights
curl -L -o weights/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
curl -L -o weights/yolo11n-seg.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt
```

**Puerto 8010**: el 8000 lo ocupa el `ai-service/` de CoppAddresd (agente IA). No usar 8000.

## Health check

```http
GET /health
```

```json
{
  "status": "healthy",
  "service": "food-ai-service",
  "version": "0.1.0",
  "timestamp_utc": "..."
}
```

## Integración con backend

`coppAddresdBack` lo consume vía `IFoodAiClient` (`GET /api/v1/foodai/health`). Config: sección `FoodAi` en appsettings del Api (`BaseUrl=http://localhost:8010`).

## Estructura

```
app/
  api/          # routers FastAPI
  core/         # config, version
  models/       # wrappers de modelos (vacío — FASE 2+)
  pipelines/    # pipeline imagen→nutrición (vacío — FASE 2+)
  schemas/      # Pydantic
  services/     # lógica (vacío — FASE 2+)
  utils/
tests/
training/       # fine-tuning futuro (checkpoints gitignoreados)
datasets/       # datasets (imágenes gitignoreadas)
docs/           # documentación del proyecto Food AI
```

## Gotchas

- Windows: no usar `uvicorn app.main:app` directo con psycopg async (ProactorEventLoop); `run_dev.py` fuerza SelectorEventLoop.
- No subir modelos, datasets ni `.env` a git.
- El backend .NET nunca depende del schema interno del servicio; contrato en `docs/api.md`.