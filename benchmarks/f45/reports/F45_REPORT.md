# F45 RESULT — Calibración final + congelación de configuración

Branch: feature/f45-final-calibration · Tests: 84/84 (9 nuevos) · Tree limpio

## Sweep completo (9 configs, cache de scores — 1 ejecución de modelos)

| Gate | Th | food-us R@1 | F101 R@1 | inv | corr | regr | abst |
|---|---|---|---|---|---|---|---|
| Top-3 | 0.65 | 42.6% | 30.9% | 371 | 276 | 66 | 29 |
| Top-3 | 0.70 | 41.7% | 31.5% | 371 | 273 | 42 | 56 |
| **Top-3** | **0.75** | **41.7%** | **32.0%** | **371** | **268** | **22** | **81** |
| Top-5 | 0.65 | 42.6% | 30.1% | 428 | 276 | 102 | 50 |
| Top-5 | 0.70 | 41.7% | 31.0% | 428 | 273 | 61 | 94 |
| Top-5 | 0.75 | 41.7% | 31.9% | 428 | 268 | 27 | 133 |
| Top-10 | 0.65 | 42.6% | 30.1% | 428 | 276 | 102 | 50 |
| Top-10 | 0.70 | 41.7% | 31.0% | 428 | 273 | 61 | 94 |
| Top-10 | 0.75 | 41.7% | 31.9% | 428 | 268 | 27 | 133 |

(v1 R@1 = 19.0% en todas las configs salvo topk=5/10+0.65: 18.5%)

## Configuración FINAL congelada (F45)
SPECIALIST_MODEL=dino_base
SPECIALIST_GROUPS=pizza,naan
SPECIALIST_THRESHOLD=0.75
SPECIALIST_GATE_TOPK=3
SPECIALIST_GATE_CONF=0.40
SPECIALIST_ENABLED=false (default — NO activado)

Regla: gate legacy (conf1 < 0.40 + pizza/naan en top-3) -> DINO-base ->
conf >= 0.75 -> promueve al top-1. Resto sin penalización. Fallback: excepción
del specialist -> ranking intacto (nunca falla el request).

## Mejor config: Top-3 + 0.75
| Sistema | R@1 | R@5 | R@10 | R@20 | R@50 |
|---|---|---|---|---|---|
| F43 base | 38.9% | 58.3% | 63.0% | 63.9% | 64.8% |
| F44 | 41.7% | 59.3% | 63.9% | 63.9% | 64.8% |
| F45 | 41.7% | 59.3% | 63.9% | 63.9% | 64.8% |
| F101 base | 32.2% | 45.3% | 49.4% | 52.8% | 53.5% |
| F101 F45 | 32.0% | 45.4% | 49.5% | 52.8% | 53.5% (-0.2) |

vs F44: regresiones 42 -> 22 (-48%), F101 -1.6 -> -0.2, R@1 y pizza
conservados. La calibración elimina la mitad de las regresiones SIN perder
la mejora.

## Regresiones explícitas (F43 correct -> F45 incorrect): 17
TODAS de una categoría única: lasagna -> pizza (14, conf 0.75-0.82) y
nachos -> pizza (3, conf 0.80-0.81). El DINO sobre-generaliza lasagna/nachos
(platos con queso/salsa) como pizza. Sin regresiones fuera de esa categoría.
22 regr totales del sweep (17 + 5 donde el base ya fallaba).

## Pizza/Naan
F38: pizza 20/20 (grupo aislado) · F44: 19-20/20 · F45: 19/20 (conf 0.75
abstiene en 1 caso marginal). La ventaja se conserva dentro del catálogo de
1.451. naan: sin ground truth en food-us/v1/F101 (no medible — heredado del
F38).

## Specialist
invoked 371/3856 (9.6%) · corr 268 · regr 22 (0.57% del total) · abst 81 ·
DINO ~256ms solo en eligible -> overhead +25ms promedio por imagen.

## Latencia
Retrieval ~100ms + grouping <1ms + DINO 256ms (9.6% de imágenes) = +25ms
promedio. p50/p95: la infraestructura de perfilado actual no mide por
percentil en este pipeline (heredado) — overhead total documentado.

## Código
- app/models/retrieval_rerank.py: regla pura (gate/umbral/promoción/fallback)
  con las constantes congeladas — TESTEABLE sin modelos.
- tests/test_retrieval_rerank.py: 9 tests (gate, umbral, promoción, regresión
  non-pizza documentada, fallback, constantes congeladas).
- scripts/f45_sweep.py: sweep reproducible (cache de scores).
- NOTA: los módulos SpecialistRouter/SpecialistShadow del F38-F40 viven en
  feature/f38/f39/f41 (rama del otro agente) — NO en la rama carlos. El
  modelo data/models/f38/pizza_naan_dino_base.pt SI está. Para F46 (shadow)
  se traerán los módulos de feature/f38.

## Producción
Intacta: retrieval off, specialist off, legacy = respuesta, frontend/backend
sin tocar, 84/84.

## Decisión
APPROVE F45 — configuración final congelada (Top-3 + 0.75): regresiones
reducidas 48% (22, categoría única lasagna/nachos->pizza), R@1 41.7
conservado, F101 casi el base (-0.2), pizza 19/20, overhead +25ms. Mejora
más segura que F44. Siguiente: F46 shadow real -> F47 escala 10.000.
