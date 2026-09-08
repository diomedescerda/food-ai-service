# F57 RESULT — Hardening + Monitoring (producción)

Branch: feature/f57-hardening-monitoring · Tests: 160/160 (8 nuevos) · Tree limpio

## Release (versión operativa)
RELEASE: pipeline=f57 catalog=5761 retrieval=multitext-v1
reranker=general-v1 specialist=dino-base-pizza-naan-v1 nutrition=mapping-v1
index=multitext-7498 — logueado en startup.

## Health checks (runtime verificado)
- Liveness (/health): detector/segmenter/classifier + pipeline + nutrition
- Readiness (/health/readiness): status=ready catalog=5761 index=True
  clip=True dino=True nutrition=True pipeline_version=f57
- Readiness NO se bloquea por el pipeline/nutrition (opcionales, fallback
  legacy seguro); sí por los modelos obligatorios.

## Métricas (/metrics, runtime verificado)
total_requests=1 new_pipeline_used=1 (el request del analyze los
incrementó). Contadores: total/success/failed, new_pipeline_used,
legacy_fallback, fallback_pipeline_error, fallback_low_confidence,
nutrition_ready/unavailable/errors, specialist_calls, portion.

## Fallbacks observables
Reasons separados (pipeline_error / nutrition_unavailable / low_confidence)
— no escondidos en una métrica. Failure drill: retrieval/DINO unavailable ->
pipeline sin specialist; nutrition unavailable -> identificación intacta.

## Consistencia
Catálogo 5.761 == índice multi-text 7.498 textos (canonical + aliases) —
verificado por test; el catálogo no cambia en operación normal.

## Excepción conocida
hamburger_011 (CORRUPT_FILE del dataset) — no es error del pipeline;
documentada como excepción de benchmark.

## Rollback (procedimiento)
FOOD_AI_RETRIEVAL_ENABLED=true -> false -> legacy exacto. Probado en F56
(etapa 0%) y F54 (OFF==ON). Sin redeploy, por flag.

## Nutrition
2.184/5.761 (37.9%) observable vía /metrics y mappings.json; tandas USDA
reanudables independientes (un alimento UNAVAILABLE -> READY se usa
automáticamente en runtime sin redeploy).

## Performance
Baseline operativo F56: p50 ~2.2-2.4s (documentado; sin SLA inventado).

## Concurrencia
1 worker + semáforo existente; 4 concurrentes validado (F54/F55/F56).

## Decisión
APPROVE F57 — PRODUCTION HARDENED: el sistema es operable (health +
readiness + metrics), observable (RELEASE + métricas + fallback reasons),
reversible (rollback por flag probado) y mantenible (nutrition coverage
independiente, cache local, cero red en inferencia). Sin cambiar el modelo.

## Después de F57 (independientes)
1. Completar ~3.577 nutrition mappings (tandas diarias).
2. Corregir casos de baja confianza con errores reales.
3. Añadir alimentos / escalar a 10K solo si el producto lo necesita.
