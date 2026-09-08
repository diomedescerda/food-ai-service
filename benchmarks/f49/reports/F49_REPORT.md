# F49 RESULT — Recall Expansion (K sweep + multi-query)

Branch: feature/f49-recall-expansion · Tests: 110/110 (6 nuevos) · Tree limpio

## K sweep (single query, embeddings existentes — sin entrenar)
| Dataset | R@50 | R@100 | R@200 | R@500 |
|---|---|---|---|---|
| food-us | 53.7% | 65.7% | 73.1% | 74.1% |
| v1 | 31.9% | 38.7% | 45.6% | 56.5% |
| Food-101 | 51.6% | 56.7% | 60.1% | 62.1% |

## Buckets A (food-us): GT fuera del Top-50
A1 (51-100): 13 · A2 (101-200): 8 · A3 (201-500): 1 · A4 (>500): 28 (26%)
v1: A4 = 108 (44%) · Food-101: A4 = 1325 (38%)

El recall del pool-200: food-us 53.7 -> 73.1 (+19.4), v1 +13.7, food101 +8.5.
El bucket A se reduce 46% -> 27% (food-us) pero el A4 (>500) sigue alto:
26-44% de los GT fuera del top-500 — el LÍMITE ES LA REPRESENTACIÓN
(textual CLIP), no el tamaño del pool.

## F49-A (single -> Top-200 -> reranker F48)
| Config | food-us | v1 | Food-101 |
|---|---|---|---|
| F48 (top-50) | 38.9% | 21.0% | 34.2% |
| F49-A sin rank | 43.5% | 16.5% | 19.9% |
| F49-A w_rank=0.15 | 38.9% | 17.7% | 24.7% |
| F49-A w_rank=0.30 | 38.9% | 18.1% | 24.8% |

## F49-B (multi-query 4 vistas -> fusión max -> reranker)
food-us 38.9 (sin rank), v1 14.9-18.1, food101 18.0-25.0 — el multi NUNCA
supera el single (las vistas no aportan recall adicional relevante).

## Hallazgo central
El reranker F48 es calibrado para el pool del top-50. Con el pool-200:
los canónicos genéricos del 51-200 (support/alias altos) desplazan los GT
(food101: 34.2 -> 24.7 — REGRESIÓN). El w_rank modera pero no recupera.
El multi-query (4 vistas) no aporta recall adicional sobre el single.

## Métricas (mejor config: F49-A sin rank, food-us)
MRR 0.543 (vs F48 0.458) · med_rank 1 · prom 63 · dem 7 · A=29 (27%)
B=68 · pool 200 · duplicate ratio 0 (grouping por canonical).

## Performance
Single top-200: retrieval 4x matmul ~8ms + reranker <1ms (food-us 108
imgs: 8s totales incluyendo CLIP visual ~80ms). Multi-query: 4x CLIP
visual (~320ms/img) + 4x matmul — el multi cuesta 4x sin ganancia.

## Producción
Intacta: retrieval off, shadow off, legacy = respuesta, 110/110.

## Decisión
REQUIRES BETTER RETRIEVAL REPRESENTATION (criterio §19 del prompt):
- El recall del pool-200 existe (A reducido 46 -> 27% food-us) pero:
  1) el reranker F48 no escala al pool-200 (regresiones v1/food101);
  2) el A4 (>500): 26-44% de los GT fuera del top-500 — el límite es la
     representación CLIP textual, NO el tamaño del candidate pool.
- El multi-query no aporta (las vistas del mismo CLIP correlacionan).
- NO se escala a 10k todavía (el R@500 del v1 56.5 es bajo).

## Siguiente (documentado, sin entrenar todavía)
1) La representación: las ALIASES del canonical como prompts adicionales en
   el índice (el GT "hamburger" rankea mal porque el índice solo tiene el
   canonical — las variantes del FNDDS como queries textuales del índice).
2) El reranker segmentado: top-50 F48 (ranking) + top-200 (recall solo si
   confianza baja) — sin mezclar los pools.
3) F50: evaluación de la representación multi-prompt (siempre sin
   entrenamiento).
