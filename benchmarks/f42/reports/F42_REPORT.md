# F42 RESULT

Branch: feature/f41-food-retrieval-engine (continuación)
Commit: (final, tree limpio) · Tests: 75/75 (3 nuevos de retrieval)

## Catálogo (recuperado del commit 8d3691b + completado)
- Fuente: USDA FNDDS (Survey), 34+39 queries de categoría — licencia CC0
- RAW: 2.572 registros · Entradas finales: 1.451 · Canónicos (genéricos): ~250
  (política anti-variantes: 'breakfast pizza with egg' -> canonical 'pizza')
- Aliases: las variantes normalizadas como aliases del canonical
- 100% con nutrition_mapping FNDDS (fdc_id + source)

## Bugs del API resueltos (diagnóstico F41 + 1 nuevo)
1. query obligatorio (400 sin él) ✓ 2. espacios -> %20 ✓ 3. pageSize=50
   (100/25 -> 400) ✓ 4. load_dotenv override=True ✓
5. NUEVO: pageNumber=0 -> 400 (el API es 1-indexed) -> page+1 ✓
6. Retry limitado (3 intentos) para los 400 intermitentes ✓

## Embeddings
- Modelo: CLIP ViT-B/32 (existente) · dim 512 · 3 templates (photo/picture/
  close-up) · 1.451 x 512 x 3 · cache en catalog/embeddings/ (3 x ~3.7MB)

## Índice
- Tecnología: numpy exacto (1.451 x 512, mean de templates, normalizado)
- Tamaño: ~3.7MB · RAM: ~6MB · construcción: <1s · FAISS documentado para 100k+
- Reconstruible desde catalog/foods.json + embeddings

## Motor (app/models/food_retrieval.py)
retrieve(image_or_crop, top_k) -> candidates [{food_id, name, score}]
top_k = 5/10/20/50 ✓ · carga una sola vez · smoke test: pizza_001 -> top-5 pizza

## Evaluación (Recall@K)
| Dataset | R@1 | R@5 | R@10 | R@20 | R@50 |
|---|---|---|---|---|---|
| food-us (108) | 38.9% | 51.9% | 57.4% | 58.3% | 64.8% |
| food-bench-v1 (247) | 25.0% | 34.3% | 38.8% | 40.7% | 45.5% |
| Food-101 (3500) | 31.5% | 42.2% | 46.3% | 49.5% | 52.7% |

Legacy top-1: 71.3% food-us — el retrieval NO lo supera (esperado: el
legacy usa el scoring curado de 38; el objetivo F41 es la COBERTURA masiva).

## Cobertura
1.451 entradas · 100% con canonical_name + aliases + text embedding +
nutrition_mapping FNDDS · visual embeddings: N/A (imágenes de referencia no
obligatorias en F41)

## Performance
Embedding text: ~45s (1.451 x 3) · Retrieval por imagen: ~100ms (CLIP visual)
+ <1ms (búsqueda numpy) · Latencia total ~100ms

## Feature flags
FOOD_AI_RETRIEVAL_ENABLED=false (default) · FOOD_AI_RETRIEVAL_SHADOW_ENABLED=false
· Con ambas off: el sistema productivo es idéntico (sin cambios en analyze)

## Producción
Legacy intacto · SpecialistRouter/Shadow sin cambios · sin integración
nutrition (F41: imagen -> alimento candidato únicamente)

## Limitaciones
- Canónicos genéricos ~250 (el FNDDS con la política anti-variantes); las
  1.451 entradas incluyen las variantes (agrupadas por canonical)
- R@K del texto directo: más débil que el clasificador legacy (esperado)
- El ranking rankea las VARIANTES (todas con canonical 'pizza') — el Top-K
  puede tener el mismo canonical repetido (agrupar en el siguiente paso)

## Decisión
APPROVE F41/F42: catálogo reproducible >=1.000, embeddings 100%, índice
funcional, Top-K recupera el alimento correcto (pizza_001 -> top-5 pizza),
R@1/5/10/20/50 medidos, legacy intacto, flags con rollback, 75/75 tests.
Sin entrenamiento adicional.

## Siguiente
1) Agrupar los candidatos por canonical en el Top-K (el ranking actual
   devuelve variantes repetidas).
2) Reranking con el SpecialistRouter/legacy para el top-1 (F45+).
3) Shadow retrieval (FOOD_AI_RETRIEVAL_SHADOW_ENABLED) con telemetría
   top1/5/10/20/50 sin cambiar la respuesta.
