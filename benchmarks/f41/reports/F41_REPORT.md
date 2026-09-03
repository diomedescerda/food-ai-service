# F41 RESULT

Branch: feature/f41-food-retrieval-engine
Commit: (final, tree limpio)
Tests: 78/78

## Estado del trabajo
El componente de diagnóstico del importador se completó parcialmente:
- Fuente elegida: USDA FNDDS (Survey) via API — CC0 ✓
- Bugs del API documentados y resueltos en el código (en el momento del test):
  1. El search exige query (400 sin él) -> query por categoría
  2. urlencode usa '+' para espacios (400) -> replace %20
  3. pageSize=100/25 -> 400; pageSize=50 -> 200 (límite real)
  4. load_dotenv(override=False) no reemplaza la key vacía del proceso -> override=True

## BLOQUEO (informativo)
El archivo scripts/import_open_catalog.py DESAPARECIÓ del disco durante la
sesión (los edits lo leyeron correctamente; el archivo fue eliminado por una
actividad externa — patrón observado en F20-F23 con la otra sesión que edita
el monorepo). El importador no completó la descarga del catálogo.

## Catálogo
Objetivo: >= 1.000 canónicos desde ~3.000 FNDDS (34 queries de categoría × 50).
NO completado (el importador no ejecutó).

## Retrieval / Índice / Evaluación
NO implementados (bloqueados por el importador).

## Feature flags / Producción
Sin cambios: FOOD_AI_RETRIEVAL_ENABLED=false (default), legacy intacto,
78/78 tests, tree limpio.

## Decisión
REQUIRES CORRECTION: el importador no completó (archivo eliminado
externamente). El diagnóstico del API queda documentado; el reintento es
mecánico (recrear el importador con los 4 fixes conocidos).

## F42 (reintento)
1) Recrear scripts/import_open_catalog.py con los 4 fixes documentados
   (query por categoría, %20, pageSize=50, override=True) + guardar el
   resultado ANTES de cualquier otra edición.
2) Normalización/dedup -> >=1.000 canónicos -> catalog/foods.json.
3) Embeddings CLIP (3 templates) cacheados + índice numpy (FAISS para la
   escala 100k+ — no instalado) -> app/models/food_retrieval.py.
4) Evaluación R@1/5/10/20/50 (food-us/v1/Food-101) + flags.
