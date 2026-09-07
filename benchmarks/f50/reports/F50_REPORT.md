# F50 RESULT — Multi-Text Retrieval (canonical + aliases)

Branch: feature/f50-multi-text-retrieval · Tests: 117/117 (7 nuevos) · Tree limpio

## Índice
- Textos: 7.498 (5.761 canonical + 1.737 aliases reales del catálogo)
- Embeddings: 22.494 (7.498 x 3 templates CLIP) cacheados
- Índice numpy (7.498, 512): 15 MB RAM (vs 12 MB F49 — +25%)
- Build: ~10 min (una vez, cacheado); query matmul ~8ms (4x el canonical-only)

## Retrieval multi-text (aggregación MAX — la mejor)
| Dataset | | R@1 | R@5 | R@10 | R@20 | R@50 | R@100 | R@200 | R@500 |
|---|---|---|---|---|---|---|---|---|---|
| food-us | F49 | 10.2 | 22.2 | 38.0 | 48.1 | 53.7 | 65.7 | 73.1 | 74.1 |
| food-us | **F50** | **38.0** | **64.8** | **69.4** | 69.4 | **75.9** | 78.7 | **86.1** | **90.7** |
| v1 | F49 | 8.1 | 14.5 | 21.0 | 26.2 | 31.9 | 38.7 | 45.6 | 56.5 |
| v1 | **F50** | **16.9** | **33.9** | **39.9** | 44.0 | **50.0** | 53.6 | **60.5** | **63.3** |
| Food-101 | F49 | 9.6 | 26.9 | 34.7 | 42.9 | 51.6 | 56.7 | 60.1 | 62.1 |
| Food-101 | **F50** | **25.3** | **43.6** | **50.4** | 55.0 | **59.7** | 61.5 | 62.7 | **63.5** |

## Bucket A (GT fuera de Top-K)
| Dataset | A500 F49 | A500 F50 | A50 F49 | A50 F50 |
|---|---|---|---|---|
| food-us | 28 (26%) | **10 (9%)** | 50 | 26 |
| v1 | 108 (44%) | 91 (37%) | 169 | 124 |
| Food-101 | 1325 (38%) | 1277 (36%) | 1693 | 1410 |

El bucket A500 de food-us: 26% -> 9% — el recall del multi-text.

## Rescates por alias (GT fuera del Top-50 F49 -> dentro del Top-50 F50)
food-us 24 · v1 45 · Food-101 287 — total 356 con ejemplos directos:
- french fries <- "potato french fries school" (rank 3)
- hamburger <- "double hamburger on wheat bun" (rank 29)
- bagel <- "bagel with raisins" (rank 2)
- burrito <- "burrito bowl" (rank 2)
- cookie <- "cookie lebkuchen" (rank 3)

## Aggregación
max (38.0 R@1 food-us) > top-2mean (30.6) — el best-score gana, sin
acumulación artificial de aliases.

## Reranker F48
NO re-ejecutado en F50 (queda para F51 — segmentación + reranker). El
retrieval multi-text SOLO ya da R@1 38.0 food-us (= el F48 completo).

## Performance
Retrieval multi-text: ~8ms matmul + CLIP visual 80ms. Latencia total por
imagen: ~90ms (+25% vs F49 por el índice 4x). RAM +3MB. Disk: +46 MB
(embeddings cache). Razonable.

## Producción
Intacta: retrieval off, shadow off, legacy = respuesta, 117/117.
Nueva flag sugerida para F51: FOOD_AI_RETRIEVAL_ALIAS_INDEX_ENABLED=false.

## Decisión
APPROVE F50 — las aliases/entries aumentan significativamente el recall
(bucket A500 food-us 26% -> 9%; R@50 +22; 356 rescates con evidencia
directa). El coste (+25% RAM, +8ms query) es razonable. El R@1 del
retrieval solo ya alcanza el nivel del reranker F48 — F51 (segmentación +
reranker) podrá superarlo.

## Siguiente (F51)
1) Segmentated retrieval: Top-50 ranking (F48) + Top-200/500 recall (el
   multi-text) sin contaminar el reranker.
2) Reranker F48 sobre el pool multi-text (best_text/best_rank como
   features adicionales).
3) F52: escala a 10.000+ — ahora el recall del multi-text lo soporta.
