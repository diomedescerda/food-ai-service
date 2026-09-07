# F52 RESULT — Nutrition Mapping Masivo (5.761 alimentos)

Branch: feature/f52-nutrition-mapping · Tests: 134/134 (10 nuevos) · Tree limpio

## Cobertura de mapping (estructura del catálogo — 0 requests)
A (fdc_id existente en el catálogo): 5.542 (96%) · B (USDA sin id): 0 ·
C (Open Food Facts): 219 (4%) · D (sin fuente): 0

## Nutrientes descargados (hoy)
- USDA FDC (food/<fdc_id> abridged, per 100g): 1.000 alimentos (límite
  diario del API alcanzado — caché reanudable, ~4.542 pendientes)
- Open Food Facts (api/v3 nutriments per 100g): 184/219
- TOTAL: 1.184 con nutrientes (20.6%) + 35 OFF sin nutrientes marcados
- Errores: 0 (120+880+219 requests, sin 429s)

## Golden foods (validados)
pizza 292 kcal/100g ✓ · hamburger 290 ✓ · rice 119 ✓ · fries 185 ✓ ·
chicken 104 ✓ · pasta 87 ✓ · sandwich 395 ✓ · bread 177 ✓ — todos
NUTRITION_READY, unidades kcal/g/mg explícitas, referencia 100g.
(naan: alias de bread — se resuelve vía el canonical bread; el canonical
"naan" no existe.)

## Muestra manual (50 aleatorios)
correct (USDA directo, fdc_id del mismo alimento): 38 · acceptable (OFF):
7 · unavailable: 5 · ambiguous: 0 — sin mappings inventados (todos provienen
del fdc_id real del catálogo o del producto OFF).

## NutritionService (runtime)
- Lookup local (nutrition/mappings.json + raw/): CERO red en inferencia.
- Estados: NUTRITION_READY / NUTRITION_UNAVAILABLE.
- Confianzas separadas: visual (pipeline) vs nutricional (0.95 USDA / 0.85
  OFF) vs porción (externa, sin combinar).
- Normalización: nutrientes_per_100g + reference_grams=100 + unidades
  explícitas (kcal/g/mg).

## Integración
- Flags: FOOD_AI_NUTRITION_ENABLED=false · NUTRITION_SHADOW_ENABLED=false
  (default; legacy = respuesta).
- Shadow: hook post-response: canonical -> nutrition lookup -> log
  (canonical, status, conf, source, cal/protein/carbs/fat, ref).
- Fallback: sin mapping -> UNAVAILABLE -> la nutrición existente sigue.

## Reanudable (límite diario USDA)
python scripts/f52_nutrition.py --priority (próximos días hasta completar
los 5.542; el índice se preserva).

## Producción
Retrieval off, nutrition off, shadow off, specialist off — legacy = respuesta,
134/134. Recognition intacto (sin cambios en el pipeline F51).

## Decisión
PARTIAL SUCCESS (con camino claro): el mapping estructural cubre el 96%
(fdc_id reales — sin inventar), la nutrición precomputada funciona con
confianza por fuente y golden validados. La cobertura de nutrientes es
20.6% hoy por el límite diario del API USDA (reanudable ~4 días). El
siguiente paso (F53) completará las tandas y añadirá confidence+fallback.
