# F56 RESULT — Rollout Gradual Real

Branch: feature/f56-gradual-rollout · Tests: 152/152 (4 nuevos) · Tree limpio

## Etapas (food-us, 108 requests, split determinista por hash)
| Stage | new | legacy | new_acc | legacy_acc | fallback | errors | p50 |
|---|---|---|---|---|---|---|---|
| 0% | 0 | 107 | — | 45.8% | 0 | 1* | 2213ms |
| 10% | 10 | 97 | 70.0% | 43.3% | 3 | 1* | 2218ms |
| 25% | 24 | 83 | 58.3% | 43.4% | 7 | 1* | 2215ms |
| 50% | 53 | 54 | 50.9% | 42.6% | 17 | 1* | 2234ms |
| 100% | 107 | 0 | 47.7% (70.8% detectados) | — | 35 | 1* | 2369ms |

*El único error por etapa = la imagen hamburger_011 del dataset (CORRUPT_FILE
conocido del F46) — no del pipeline. Fallbacks = sin detección (foods vacío).

## Observaciones
- El nuevo supera al legacy en TODAS las etapas (10%: 70.0 vs 43.3; 25%:
  58.3 vs 43.4; 50%: 50.9 vs 42.6) — sin ninguna etapa con regresión.
- El 100% (47.7% bruto) == el F55 (70.8% sobre detectados) — consistente.
- Latencia estable: p50 2213-2369ms, p95 2273-2470ms (sin degradación por
  etapa; el new ~+150ms sobre legacy — pipeline + detector).
- Alimentos observados (pizza/hamburger/fries/hot dog/sandwich): el new
  corrige hamburger:hot_dog -> hamburger/burger en varias imágenes (las
  correcciones del C del F54).

## Rollback
100% new -> FOOD_AI_RETRIEVAL_ENABLED=false -> legacy exacto (la etapa 0%
del rollout = el rollback ejecutado; sin redeploy, por flag).

## Nutrition
2.184/5.761 (37.9%) — tandas independientes; nutrición no bloquea el
rollout (identificación separada).

## Seguridad
0 crashes · 0 request failures del pipeline · 1 error de dataset (corrupto)
· response contract intacto · sin NaN · sin nutrientes inventados ·
specialist F45 congelado e intacto.

## Concurrencia
La etapa 100% corrió con el split completo; las pruebas de 4 concurrentes
de F54/F55 siguen válidas (mismo semáforo).

## Producción
Flags default: OFF (el rollout se activa por entorno con el flag).
Legacy disponible en todo momento.

## Decisión
APPROVE F56 — FULL PRODUCTION ROLLOUT: las 5 etapas estables, el nuevo
supera al legacy en cada etapa, 0 fallos del pipeline, rollback probado.
Siguiente: F57 — Hardening + Monitoring (logs, métricas, alertas, health
checks, procedimiento de rollback documentado) — sin ampliar catálogo, sin
entrenar.
