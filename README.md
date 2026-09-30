# FoodAI Service

> **Contexto compartido del workspace**: lee `../.coppadresd-context/` (estado actual, arquitectura, decisiones, ítems abiertos, runbook) ANTES de modificar código. Última verificación: 2026-09-30. Consumido SOLO por `coppAddresdBack` (FoodAiController); prod verificado healthy 2026-09-30. Setup en este README; contrato API en `docs/api.md`.

Servicio de IA de análisis de alimentos del ecosistema CoppAddresd: detección, clasificación (reconocimiento de **5.761 alimentos**), estimación de porción y nutrición. Python 3.11+ / FastAPI / CLIP + DINO.

**Estado actual: FASE 57 — PRODUCTION HARDENED.** El sistema está desplegado y operando con el pipeline de 5.761 alimentos como resultado principal (activado por flag), con legacy como red de seguridad, monitoreo operativo (health/readiness/metrics) y rollback por flag.

---

## 1. Pipeline de reconocimiento (F51+)

```
Imagen
  ↓
Detector (YOLO hybrid + DINO fallback) → crops
  ↓
CLIP ViT-B/32 image embedding
  ↓
Multi-text retrieval (canonical + aliases, 7.498 textos × 3 templates)
  ↓
Canonical grouping (dedup por canonical, best_text)
  ↓
General reranker (retrieval + support + alias_count, determinista)
  ↓
DINO-base specialist pizza/naan (threshold 0.75, gate top-3, conf legacy < 0.40)
  ↓
Food canonical final
  ↓
NutritionService (lookup local, per 100 g, cero red)
  ↓
BasicPortionEstimator / AdvancedPortionEstimator
  ↓
DecisionPolicy (confidence / fallback legacy)
```

La red neuronal **no tiene 5.761 clases**: el catálogo vive en texto + embeddings + índice numpy. Agregar alimentos = agregar registros + embeddings, no entrenar.

## 2. Comandos

```bash
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest        # tests (160+)
.\.venv\Scripts\python run_dev.py       # dev server puerto 8010 (Windows: selector loop)
```

**Modelos** (primera vez):

```bash
mkdir weights
curl -L -o weights/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
curl -L -o weights/yolo11n-seg.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt
```

**Puerto 8010**: el 8000 lo ocupa el `ai-service/` (agente IA). No usar 8000.
**Windows**: no usar `uvicorn app.main:app` directo (ProactorEventLoop incompatible con psycopg async); `run_dev.py` fuerza SelectorEventLoop. Para pruebas sin reload: `scripts/f46_server.py` (puerto por env `PORT_F46`).

## 3. Configuración (flags de feature)

Todas en `.env` (o variables de entorno). **Defaults = legacy puro** (producción segura):

| Flag                               | Default | Efecto                                                                        |
| ---------------------------------- | ------- | ----------------------------------------------------------------------------- |
| `FOOD_AI_RETRIEVAL_ENABLED`        | `false` | `true` → la respuesta usa el pipeline 5.761 (active mode)                     |
| `FOOD_AI_RETRIEVAL_SHADOW_ENABLED` | `false` | `true` → ejecuta el pipeline en shadow (telemetría, respuesta legacy intacta) |
| `FOOD_AI_NUTRITION_ENABLED`        | `false` | `true` → nutrición mapeada en la respuesta                                    |
| `FOOD_AI_NUTRITION_SHADOW_ENABLED` | `false` | `true` → shadow de nutrición (log, sin tocar la respuesta)                    |
| `FOOD_AI_CONFIDENCE_ENABLED`       | `false` | `true` → el threshold de confianza visual puede rechazar (LOW_CONFIDENCE)     |
| `FOOD_AI_MIN_VISUAL_CONFIDENCE`    | `0.20`  | umbral usado solo si CONFIDENCE_ENABLED=true                                  |
| `FOODAI_USDA_API_KEY`              | —       | API key USDA FoodData Central (descargas de nutrición, nunca imprimir)        |

**Rollback**: `FOOD_AI_RETRIEVAL_ENABLED=false` → legacy exacto, sin redeploy.

## 4. Endpoints

| Endpoint                | Descripción                                                                          |
| ----------------------- | ------------------------------------------------------------------------------------ |
| `POST /analyze`         | Análisis completo (multipart: `analysis_id`, `image`)                                |
| `GET /health`           | Liveness: detector/segmenter/classifier + pipeline + nutrition                       |
| `GET /health/readiness` | Readiness: catalog_size, index/clip/dino/nutrition loaded, pipeline_version          |
| `GET /metrics`          | Métricas agregadas en memoria (requests, pipeline, fallbacks, nutrition, specialist) |

`GET /api/v1/foodai/health` (backend): el backend .NET lo consume vía `IFoodAiClient` (`BaseUrl=http://localhost:8010`).

## 5. Catálogo (5.761 alimentos)

- **5.761 canonical foods** / 9.067 entries / 1.737 aliases (schema_version 3).
- Fuentes: **USDA FDC** (FNDDS + SR Legacy, CC0) + **Open Food Facts** (ODbL).
- Archivos: `catalog/foods.json`, `catalog/foods_legacy_1451.json` (snapshot de preservación), `catalog/embeddings/` (embeddings CLIP del catálogo), `catalog/embeddings/multitext/` (7.498 textos × 3 templates = 22.494 embeddings, índice multi-text).
- Política de canonicalización: los platos preparados colapsan al genérico (`"beef cheeseburger with bacon" → "hamburger"`); los ingredientes quedan a nivel específico USDA. Nunca fusionar alimentos realmente distintos.
- Regeneración: `python scripts/f47_import_catalog.py` (reanudable, cache de OFF por query) + `python scripts/f42_retrieval.py --embeddings`.

## 6. Nutrición (F52–F54)

- **2.184 / 5.761 con nutrientes (37.9%)** y creciendo: tandas USDA diarias (~1.000 requests/día, caché reanudable) + Open Food Facts.
- `nutrition/mappings.json` + `nutrition/raw/`: lookup **100 % local, cero red en inferencia**.
- Normalización: `nutrients_per_100g` + `reference_grams=100` + unidades explícitas (kcal/g/mg).
- Estados: `NUTRITION_READY` / `NUTRITION_UNAVAILABLE` — nunca se inventan nutrientes.
- Confianza nutricional por fuente: 0.95 USDA / 0.85 OFF.
- Continuar tandas: `python scripts/f52_nutrition.py --priority [--limit N] [--off]`.
- Los alimentos `UNAVAILABLE → READY` se usan automáticamente en runtime, sin redeploy.

## 7. Confidence / Fallback (F53)

Tres confianzas **separadas** (nunca combinadas): `visual_confidence` (retrieval score), `nutrition_confidence`, `portion_confidence`.

Estados de decisión (`app/models/decision.py`):
`NEW_RESULT_READY` · `NEW_RESULT_NUTRITION_UNAVAILABLE` · `NEW_RESULT_LOW_CONFIDENCE` · `LEGACY_FALLBACK`

Reglas clave:

- La **identificación y la nutrición son independientes**: un alimento identificado sin nutrientes NO se convierte en unknown.
- Fallback reasons observables: `pipeline_error` / `nutrition_unavailable` / `low_visual_confidence` / `nutrition_error`.
- Cualquier excepción del pipeline experimental → legacy, sin romper el request.

## 8. Specialist (congelado, F45)

```
SPECIALIST_MODEL=dino_base · SPECIALIST_GROUPS=pizza,naan
SPECIALIST_THRESHOLD=0.75 · SPECIALIST_GATE_TOPK=3 · SPECIALIST_GATE_CONF=0.40
```

No crear nuevos especialistas sin evidencia independiente. El DINO se ejecuta solo cuando el gate se abre (conf legacy < 0.40 + pizza/naan en top-3 del legacy).

## 9. Benchmark (resultados medidos)

| Fase                          | Métrica                   | Resultado                                           |
| ----------------------------- | ------------------------- | --------------------------------------------------- |
| F48 reranker (catálogo 1.451) | food-us R@1               | 38.9 %                                              |
| F50 multi-text retrieval      | food-us R@50 / R@500      | 75.9 % / 90.7 %                                     |
| F51 pipeline integrado        | food-us R@1               | 56.5 %                                              |
| F54 shadow runtime (107)      | nuevo vs legacy           | **56.1 % vs 43.9 %** (+12.2, ratio corr/regr 5.3:1) |
| F55 active (sobre detectados) | nuevo vs legacy           | **70.8 % vs 65.3 %**                                |
| F56 rollout gradual           | nuevo vs legacy por etapa | nuevo gana en todas (10 %: 70.0 vs 43.3)            |
| F57 runtime                   | p50 / p95                 | ~2.2–2.4 s / ~2.4 s                                 |

Reportes por fase en `benchmarks/f{41..57}/reports/`.

## 10. Tests

```bash
.\.venv\Scripts\python -m pytest        # 160+ tests, sin API keys (modelos fake/mocks)
```

Cubren: clasificación, detección, porción, nutrición, pipeline 5.761, reranker general, grouping multi-text, specialist, decision policy, shadow, rollout, hardening, consistencia catálogo/índice.

## 11. Scripts por fase

| Script                            | Fase    | Propósito                                                     |
| --------------------------------- | ------- | ------------------------------------------------------------- |
| `f42_retrieval.py`                | F42     | Embeddings CLIP del catálogo + índice + eval R@K              |
| `f43_rerank.py` / `f44_rerank.py` | F43–F44 | Canonical grouping + rerank specialist                        |
| `f45_sweep.py`                    | F45     | Barrido de calibración del specialist (gate/threshold)        |
| `f46_server.py` / `f46_client.py` | F46     | Server sin reload + cliente multipart (shadow real)           |
| `f47_import_catalog.py`           | F47     | Importador masivo (FNDDS + SR Legacy + OFF, cache reanudable) |
| `f48_rerank.py`                   | F48     | Reranker general (features por canonical)                     |
| `f49_recall.py`                   | F49     | K sweep + multi-query (vistas)                                |
| `f50_multitext.py`                | F50     | Índice multi-text (canonical + aliases)                       |
| `f51_integration.py`              | F51     | Benchmark del pipeline integrado                              |
| `f52_nutrition.py`                | F52/F54 | Descarga de nutrientes (tandas reanudables)                   |
| `f56_rollout.py`                  | F56     | Rollout gradual por etapas (split determinista)               |

## 12. Estructura

```
app/
  api/          # routers: analyze, health (+ readiness, metrics)
  core/         # config (flags), version
  models/       # wrappers de modelos y lógica central
    food_pipeline.py      # pipeline 5.761 (F51)
    food_retrieval.py     # índice + retrieval
    retrieval_rerank.py   # grouping multi-text + reranker general + specialist gate
    nutrition_service.py  # lookup local de nutrición
    decision.py           # política de confianza/fallback
    specialist_router.py  # specialist DINO pizza/naan (F38)
    specialist_shadow.py  # shadow del specialist (F39)
    retrieval_shadow.py   # shadow del retrieval (F46)
    zero_shot_classifier.py / detector_based_classifier.py / hybrid_detector.py ...
    basic_portion_estimator.py / advanced_portion_estimator.py / depth_anything_estimator.py
  schemas/      # Pydantic (analyze, health)
catalog/          # foods.json + embeddings (catálogo e índice multi-text)
nutrition/        # mappings.json + raw/ (nutrientes precomputados)
benchmarks/       # reportes por fase + caches de features
scripts/          # herramientas de fases (importadores, benchmarks, clientes)
datasets/         # food-us, food-bench-v1, food101-subset (imágenes gitignoreadas)
docs/             # documentación del proyecto
tests/            # 160+ tests
```

## 13. Gotchas

- Windows: `run_dev.py` (SelectorEventLoop); los procesos `Start-Process` mueren al terminar el comando del shell → E2E completo en un solo comando.
- Logs del servicio van a **STDERR** (no stdout).
- El backend .NET nunca depende del schema interno; contrato en `docs/api.md`.
- `appsettings.json` del backend está gitignoreado; los `.env` del servicio también. No subir modelos, datasets, embeddings grandes ni `.env` a git.
- El API USDA tiene límite ~1.000 requests/día (retry/backoff/429); el Open Food Facts es intermitente (503/401) — siempre cache + reanudable.
- Dataset `hamburger_011` (food-us) es `CORRUPT_FILE` conocido — no es error del pipeline.
- El otro agente/sesión paralela puede mover ramas/archivos del repo — recuperar trabajo con `git log --all` / `git reflog`; los componentes F38–F40 viven en sus ramas feature (no en `carlos`).

## 14. Historial de fases (resumen)

- **F16–F22**: clasificación CLIP zero-shot, benchmark v1, catálogo 38 → USDA, candidates descriptivos, E2E 59.3 %, multi-instancia.
- **F23**: PRODUCTION READY del legacy (concurrencia, semáforo, fallbacks).
- **F25–F40**: investigación de especialistas (prototipos → confusion groups → DINO-base pizza/naan, threshold 0.75 — congelado).
- **F41–F42**: catálogo masivo (1.451 → FNDDS), retrieval CLIP.
- **F43–F44**: canonical grouping + specialist rerank (food-us 41.7 %).
- **F45**: calibración final del specialist (gate top-3 + 0.75) — configuración congelada.
- **F46**: shadow real del retrieval (invariancia verificada).
- **F47**: catálogo 4× (5.761: FNDDS + SR Legacy + OFF), embeddings + índice.
- **F48**: reranker general determinista (support + alias + specialist) — recupera el Top-1.
- **F49**: recall expansion (K sweep; límite = representación, no pool).
- **F50**: multi-text retrieval (canonical + aliases) — recall 90.7 % @500.
- **F51**: pipeline integrado único (`food_pipeline.py`), R@1 food-us 56.5 %.
- **F52–F54**: nutrición masiva (2.184 ready), shadow final (nuevo 56.1 % vs legacy 43.9 %).
- **F55–F56**: active rollout + rollout gradual 0→100 % (nuevo gana en todas las etapas).
- **F57**: hardening (readiness, metrics, RELEASE versioning, rollback por flag).

## 15. Próximos pasos (independientes, sin investigación ML)

1. Completar las ~3.577 descargas de nutrición (tandas diarias USDA).
2. Corregir casos de baja confianza basándose en errores reales.
3. Añadir alimentos o escalar a 10K/100K **solo si el producto lo necesita** (operación de catálogo + embeddings, sin reentrenar).
