# F47 RESULT — Escalado del catálogo (sin reentrenamiento)

Branch: feature/f47-scale-catalog-10k · Tests: 97/97 (8 nuevos) · Tree limpio

## Catálogo
| Métrica | Valor |
|---|---|
| canónicos | 5.761 (4.0x del 1.451) |
| entries | 9.067 |
| aliases | 1.737 |
| raw total | 39.369 (FNDDS 5.428 + SR Legacy 7.793 + OFF 26.148) |
| fuentes | USDA FNDDS (CC0), USDA SR Legacy (CC0), Open Food Facts (ODbL) |
| legacy 1.451 | preservado + verificado (test_legacy_backcompat: canónicos exactos o como canónico más específico) |
| smoke | pizza/hamburger/naan/rice/fries/chicken/pasta/sandwich: TODOS presentes (naan restaurado como alias de bread vía legacy-preserve) |

El objetivo estricto >=10.000 canónicos NO se alcanzó: la política de
canonicalización del F47 (no variantes hiper-específicas + alimentos
razonables) limita los canónicos reales del USDA+OFF a ~5.8k. El 10k
requeriría el nivel variante-preparado (FNDDS "cheeseburger double" como
canónico — contradice la política) o el OFF a gran escala (inestable:
503s/401s). DECISIÓN documentada: catálogo 4x con calidad; F48 atacará el
ranking (no el conteo).

## Embeddings
CLIP ViT-B/32 (existente, sin entrenar): 3 templates x 5.761 = 3 npy
(5.761, 512) float32 (~11.8 MB c/u) + metadata version f42-1. Cobertura 100%.

## Índice
NumPy exacto (5.761 x 512), build ~5s, RAM ~12 MB, query matmul ~2ms.
Latencia eval (CLIP visual + matmul): food-us 108 imgs en 9s (~83ms/img).

## Accuracy (comparación 1.451 -> 5.761)
| Dataset | R@1 | R@5 | R@10 | R@20 | R@50 |
|---|---|---|---|---|---|
| food-us 1.451 | 38.9% | 51.9% | 57.4% | 58.3% | 64.8% |
| food-us 5.761 | 10.2% | 22.2% | 38.0% | 48.1% | 53.7% |
| v1 1.451 | 19.0% | 29.4% | 33.9% | 36.3% | 37.1% |
| v1 5.761 | 8.7% | 16.9% | 26.1% | 32.9% | 38.5% |
| F101 1.451 | 32.2% | 45.3% | 49.4% | 52.8% | 53.5% |
| F101 5.761 | 9.5% | 26.0% | 33.9% | 42.0% | 50.4% |

Degradación esperada por el espacio 4x (más canónicos compiten en el
top-1). NO se reduce el catálogo — el problema del ranking se documenta
para F48 (general reranker).

## Reproducibilidad
scripts/f47_import_catalog.py: list API USDA (paginado completo, sin
query) + OFF search por categorías con cache incremental por query
(catalog/off_raw_cache.json — reanudable tras cuelgues del OFF).
catalog/foods_legacy_1451.json: snapshot del catálogo 1.451 (preservación).
Regeneración: python scripts/f47_import_catalog.py && python
scripts/f42_retrieval.py --embeddings.

## Producción
Intacta: retrieval off, shadow off, specialist congelado (0.75/top-3/0.40),
legacy = respuesta, flags sin cambios, 97/97.

## Decisión
F47 APROBADO PARCIAL: catálogo 4x (5.761 canónicos) con calidad y
reproducibilidad; pipeline (catálogo -> embeddings -> índice -> retrieval)
funciona con el espacio mayor; 1.451 preservados; sin entrenar. El 10k
estricto se pospone: requiere nivel variante (decisión de política) o
fuentes estables adicionales. Siguiente: F48 general reranker para 10K
(ranking, no conteo).

## Siguiente
1) F48: general reranker (retrieval + specialist + señales) para recuperar
   el ranking con 5.7k+.
2) El 10k: re-evaluar la política de canonicalización (nivel variante) o
   el OFF con paginado estable.
3) F49: nutrition mapping masivo (food_id -> USDA/FDC).
