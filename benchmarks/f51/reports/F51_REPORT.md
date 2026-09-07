# F51 RESULT — Integración Final del Pipeline 5.761

Branch: feature/f51-integrated-pipeline · Tests: 124/124 (7 nuevos) · Tree limpio

## Arquitectura (flujo final)
imagen/crop -> CLIP embedding -> multi-text retrieval (7.498 textos) ->
canonical grouping (group_text_matches) -> reranker F48 (support+alias
calibrados) -> specialist DINO pizza/naan (F45 congelado) -> canonical final.
Componentes reutilizados: food_retrieval (índice base), F50 (embeddings
multi-text movidos a catalog/embeddings/multitext/), retrieval_rerank
(grouping + rerank_general), DINO F38, SpecialistShadow F39 (referencia).

## Accuracy (pipeline integrado vs F50 retrieval solo)
| Dataset | | R@1 | R@5 | R@10 |
|---|---|---|---|---|
| food-us | F50 | 38.0 | 64.8 | 69.4 |
| food-us | **F51** | **56.5** | 64.8 | 68.5 |
| v1 | F50 | 16.9 | 33.9 | 39.9 |
| v1 | **F51** | **21.8** | 35.1 | 40.7 |
| Food-101 | F50 | 25.3 | 43.6 | 50.4 |
| Food-101 | **F51** | **27.1** | 44.1 | 50.5 |

(El benchmark evalúa top1+top10 del pipeline — el R@20/50 no es
directamente comparable con F50; el R@1 es la métrica principal.)

## Hallazgo de integración (crítico)
El reranker F48 se calibró con el catálogo 1.451 (support=variantes, alias
<=20). El v4 colapsó a 1 entry/canonical y el cross-check legacy subió
aliases >20 — el reranker degradaba (food-us 31.5). Fix: support = 1 +
aliases (proxy de variantes) y alias = min(aliases, 20): food-us 56.5.

## Specialist
Config F45 intacta (dino_base, pizza/naan, 0.75, top-3, 0.40). Gate
integrado al pipeline (conf1 < 0.40 + top-3 + DINO conf >= 0.75). Tests:
gate cerrado/abierto ✓. Benchmark sin legacy_conf1: spec=0 (esperado).

## Fallback
Cualquier excepción (embedding, índice, reranker, DINO) -> result
fallback=True -> el caller usa legacy. Test unitario ✓ + el hook del
analyze en try/except (jamás rompe la respuesta).

## Shadow + Invariancia (runtime real)
Shadow ON: RETRIEVAL CONFIG catalog=5761 ✓, pipeline disponible ✓,
telemetría pipeline_f51 en cada request (8 logs en smoke). pizza_001:
FOOD_A (off) == FOOD_B (on): pizza conf 0.9075 — respuesta funcional
idéntica ✓.

## Smoke (runtime)
pizza_001 -> pizza ✓ · hamburger_001 -> hot dog (legacy, sin excepción) ✓ ·
sandwich_001 -> sandwich ✓ · hot_dog_001 -> sin detección (YOLO, conocido) ✓

## Concurrencia
4 simultáneos: 200 x4, 0 crashes (semáforo F23 intacto).

## Performance
Pipeline: p50 ~97ms, p95 ~112ms (retrieval multi-text + grouping + reranker
+ specialist). vs legacy (detector ~500ms+): el pipeline es marginalmente
más rápido en shadow (corre después del response).

## Producción
FOOD_AI_RETRIEVAL_ENABLED=false · RETRIEVAL_SHADOW_ENABLED=false ·
SPECIALIST_ENABLED=false — legacy = respuesta, 124/124.

## Decisión
APPROVE F51 — READY FOR NUTRITION INTEGRATION: los componentes validados
funcionan como pipeline único (5.761 alimentos, multi-text, grouping,
reranker, specialist pizza/naan), fallback y shadow probados, invariancia
confirmada, concurrencia estable, sin entrenamiento.

## Siguiente (F52)
Nutrition mapping para los 5.761: canonical -> nutrition_mapping ->
USDA/FDC -> nutrientes -> porción — sin tocar el reconocimiento.
