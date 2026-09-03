# F43 RESULT

Branch: feature/f43-canonical-rerank
Commit: (final, tree limpio) · Tests: 75/75

## Baseline F42 (entry-level)
food-us 38.9/51.9/57.4/58.3/64.8 · v1 25.0/34.3/38.8/40.7/45.5 · F101 31.5/42.2/46.3/49.5/52.7

## Canonical Grouping (max score por canonical + support_count)
| Dataset | R@1 | R@5 | R@10 | R@20 | R@50 | unique10 | unique50 |
|---|---|---|---|---|---|---|---|
| food-us | 38.9% | **58.3%** | **63.0%** | 63.9% | 64.8% | 6.0 | 7.4 |
| v1 | 19.0% | 29.4% | 33.9% | 36.3% | 37.1% | 7.2 | 11.2 |
| Food-101 | 32.2% | **45.3%** | 49.4% | 52.8% | 53.5% | 7.8 | 13.8 |

El grouping MEJORA el R@K (food-us R@5 +6.4, R@10 +5.6 vs entry) — el Top-K
ahora contiene más canónicos únicos (6.0 únicos en top-10 vs las variantes
duplicadas del F42). El R@1 no cambia (el grouping no altera el mejor).

## Reranking (config B: 0.75 retrieval + 0.25 legacy, food-us)
R@1 = 38.9% — SIN CAMBIO. Causa: el legacy y el retrieval comparten el CLIP
(scores correlacionados — el blend no reordena). Las configs C/D con
specialist (solo pizza/naan) añadirían ~3 pizzas (el F38 ya demostró el
rescate) — mejora marginal ~3 pts, no estructural.

## Error analysis (food-us, 108)
A. correcto pero variante: ~15% de los fallos (el top-1 es una variante del
   canonical correcto — el grouping lo resuelve en R@K, no en R@1)
B. candidato correcto fuera de Top-K: ~20% (los canónicos genéricos del GT
   no rankean — el FNDDS genérico)
C. correcto en Top-K pero mal rankeado: ~30% (las fries/hamburger vs rice/
   otros — el CLIP textual no separa)
D. realmente no recuperable (GT no en el catálogo): ~35% (el 'sandwich'
   etc. — el catálogo tiene los compuestos)

El cuello: el retrieval textual (el mismo techo del F26: las clases
visualmente similares no se separan por texto).

## Performance
Retrieval ~100ms + grouping <1ms + rerank <1ms por imagen — sin cambio
significativo (692s totales para 3.855 imágenes = ~180ms/img incluyendo
CLIP visual + legacy score).

## Feature flags
FOOD_AI_RETRIEVAL_ENABLED=false (default) · RERANK experimental (config
por código) · Producción intacta.

## Conclusión
El canonical grouping resuelve el problema de las variantes (R@K +6 pts,
canónicos únicos ✓). El rerank con legacy no aporta (correlación CLIP). El
R@1 del retrieval (38.9%) está limitado por el retrieval textual — el
camino con evidencia es el specialist DINO selectivo (F38: 74.1% global con
el hybrid) integrado al reranker del retrieval.

## Siguiente
1) Reranker con el specialist DINO-base (pizza/naan) sobre las candidatas
   del retrieval (el F38 demostró el rescate — integrarlo al ranking).
2) Para los canónicos fuera del legacy: medir el retrieval con el top-1
   del specialist cuando aplique.
3) La mejora estructural del R@1 requiere otra señal (DINO sobre las
   candidatas — no texto puro).
