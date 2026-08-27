# Arquitectura — Food AI dentro de CoppAddresd

## Visión

Módulo de análisis nutricional por fotografía dentro del ecosistema CoppAddresd. Pipeline por etapas (nunca un solo modelo imagen→calorías):

```
imagen → detección → segmentación → clasificación → porción → nutrition DB → cálculo determinista → resultado
```

El cálculo nutricional es determinista (BD de nutrición); la IA nunca inventa macros.

## Estado actual (FASE 0)

| Capa | Estado |
|---|---|
| `coppAddresdBack` (src/CoppAddresd.Api) | `GET /api/v1/foodai/health` — probe backend → FoodAI |
| `food-ai-service/` (nuevo, Python/FastAPI) | `GET /health` en puerto 8010 |
| Frontend | Sin UI aún — se integrará en `coppaddresd-front/` (Next.js existente) |
| PostgreSQL | Compartido con CoppAddresd (5432, db `coppaddresd`); sin tablas Food AI aún |

## Integración backend (Clean Architecture)

- `IFoodAiClient` — `src/CoppAddresd.Application/Interfaces/` (contracto, sin HTTP).
- `FoodAiClient` — `src/CoppAddresd.Infrastructure/Services/` (HttpClient, timeout 10 s, nunca lanza en health).
- `FoodAiSettings` — `src/CoppAddresd.Application/Common/`, sección `FoodAi` (BaseUrl `http://localhost:8010`).
- `FoodAiController` — `src/CoppAddresd.Api/Controllers/`, `GET /api/v1/foodai/health` (AllowAnonymous, sin datos sensibles).
- DI en `ApplicationServiceExtensions` + `CorrelationIdDelegatingHandler` (patrón idéntico a `AiServiceClient`).

## Servicio Python (food-ai-service)

Desacoplado del backend. Responsable solo de inferencia/procesamiento. Estructura modular `app/{api,core,models,pipelines,schemas,services,utils}` para que cada componente de IA sea reemplazable (`IFoodDetector`, `IPortionEstimator`, … — FASE 2+).

**Puerto 8010**: el 8000 pertenece al `ai-service/` (agente IA chat de CoppAddresd).

## Decisiones registradas

- [2026-08-27] Integrar como sub-proyecto `food-ai-service/` en vez de reusar `ai-service/` (stacks distintos; evita acoplar chat LangGraph con visión).
- [2026-08-27] No crear frontend nuevo: la UI de Food AI irá en `coppaddresd-front/` cuando corresponda.
- [2026-08-27] No duplicar PostgreSQL: food-ai usará la BD compartida (schema propio `foodai.` cuando existan tablas).
- [2026-08-27] Health público sin clave interna (no expone datos); endpoints de análisis futuros sí usarán el canal `X-Internal-Key` del patrón existente.

## Stack objetivo

Backend .NET 10 (existente) · Python 3.12+/FastAPI · PyTorch/Ultralytics (FASE 2+) · PostgreSQL 16 · Docker (compose raíz) · AWS (posterior).