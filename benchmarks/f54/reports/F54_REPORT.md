# F54 RESULT — Completar Nutrition Coverage + Shadow Final

Branch: feature/f54-nutrition-shadow-final · Tests: 142/142 · Tree limpio

## Nutrition coverage
| | |
|---|---|
| total canónicos | 5.761 |
| NUTRITION_READY | **2.184 (37.9%)** (USDA 2.000 + OFF 184) |
| pendientes | ~3.577 (tandas reanudables, límite USDA ~1.000/día) |
| requests hoy | 1.000 (400+600), 0 errores, 0 NaN/negativos |
| unidades | kcal / g / mg (explícitas, referencia 100g) |

## Shadow final (food-us completo por API real, 107 imágenes)
| | legacy | nuevo |
|---|---|---|
| top-1 (vs GT) | 43.9% (47) | **56.1% (60)** |
| A ambos correctos | 44 (41%) |
| B legacy ok / nuevo mal | **3 (2.8%)** |
| C legacy mal / nuevo ok | **16 (15%)** |
| D ambos mal | 44 (41%) |

El nuevo supera al legacy en +12.2 pts con ratio corrección/regresión 5.3:1.
Nutrition READY en runtime: 91.6% (98/107 de los alimentos reconocidos).

## Invariancia pública
Shadow OFF vs ON: pizza 0.9075 IDÉNTICOS (food + confidence) ✓ — los hooks
(retrieval/nutrition/decision) corren después del response, solo loguean.

## Concurrencia
4 simultáneos con shadow completo: 200 x4, 0 crashes, 0 deadlocks.

## Muestra 50 nuevos (calidad)
41 correct (USDA directo) + 9 acceptable (OFF) + 0 unavailable + 0
ambiguous — sin mappings inventados.

## Runtime (logs verificados)
retrieval shadow: 107 logs (pipeline_f51) · nutrition shadow: 72+ logs ·
decision shadow: 107 logs (decision + fallback_reason) · catalog_size=5761.

## Error handling
retrieval/DINO error -> pipeline fallback -> legacy; nutrition error ->
identificación intacta (DecisionPolicy F53). Sin romper requests.

## Performance
Shadow por imagen ~2.5s (pipeline 97ms + nutrition lookup local
despreciable + DINO en eligible); nutrition lookup: cero red.

## Producción
TODO OFF: RETRIEVAL_ENABLED, NUTRITION_ENABLED, CONFIDENCE_ENABLED,
RETRIEVAL_SHADOW, NUTRITION_SHADOW (default) — legacy = respuesta.

## Decisión
APPROVE F54 — READY FOR ACTIVE ROLLOUT: shadow completo estable en runtime
real (107 peticiones, 0 errores), nuevo > legacy (+12.2 pts, 3 regresiones
únicamente), invariancia confirmada, nutrición 37.9% y creciendo (tandas
reanudables), fallback probado. El F55 puede activar el rollout controlado
con el rollback a legacy disponible.

## Siguiente (F55)
Active rollout controlado: FOOD_AI_RETRIEVAL_ENABLED=true en entorno
controlado + shadow del nutrition + legacy como fallback. Sin ampliar
catálogo, sin entrenar.
