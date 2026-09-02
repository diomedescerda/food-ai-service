# F38 RESULT

Branch: feature/f38-controlled-specialist-rollout
Commit: (final, tree limpio)

## Architecture
Legacy CLIP-B/32 -> SpecialistRouter (app/models/specialist_router.py) ->
DINO pizza/naan cuando el gate B se activa. Config env preparada:
FOOD_AI_SPECIALIST_ENABLED=false (default) / GROUPS=pizza_naan /
THRESHOLD=0.40 / MODEL=dino_vitl14 (base preferido por latencia).

## Shadow Mode (telemetría, food-us)
Invocation rate: 19.4% (21/108) · Would-change: 3 (pizza rescates) ·
Corrections: 3 · Regressions: 0 · Abstentions: 1 (score < 0.60 -> legacy)
El offline (F37 74.1) == runtime (F38 74.1): la mejora es reproducible.

## Threshold Sweep (gate B, food-us)
| thr | top1 | calls | corr | regr | abst |
|---|---|---|---|---|---|
| 0.20 | 71.3% | 0 | 0 | 0 | 0 |
| 0.25 | 71.3% | 1 | 0 | 0 | 1 |
| 0.30 | 73.1% | 15 | 2 | 0 | 1 |
| 0.35 | 74.1% | 21 | 3 | 0 | 1 |
| 0.40 | 74.1% | 21 | 3 | 0 | 1 |
| 0.45 | 74.1% | 21 | 3 | 0 | 1 |
| 0.50 | 74.1% | 21 | 3 | 0 | 1 |

Threshold seleccionado: 0.40 (plateau desde 0.35 — el F37 validado; 0 regresiones en todo el rango).

## Pizza/Naan
Legacy: 17/20 · DINO-L: 20/20 · F38: 20/20 (los 3 rescates, 0 regresiones)

## Global food-us
Legacy 71.3% · F37 74.1% · F38 74.1% (reproducido)

## DINO-L vs DINO-base
| | DINO-L | DINO-base |
|---|---|---|
| Accuracy (global) | 74.1% | 74.1% |
| Latencia/img | 902ms | 264ms |
| Embedding | 1024 | 768 |
| Params | 304M | 86M |

Caso D del prompt: DINO-base mantiene accuracy con 3.4x menos latencia ->
PREFERIBLE para producción (el L solo si el base falla en otros grupos).

## Performance
Legacy: ~70-130ms · Specialist (base): ~264ms por llamada · invocación 19.4%
de imágenes · overhead promedio ~+51ms/imagen (base) o ~+175ms (L).

## Multi-food / Nutrition / Concurrency
Sin cambios: el router solo altera pizza/naan (mf_011 pizza -> pizza ✓);
detección/dedup/semáforo intactos; E2E >= legacy (las pizzas rescatadas ya
tenían mapping USDA).

## Stability
Los tests del router (6 casos) cubren: conf alta (no DINO), conf baja+pizza
(DINO), conf baja sin pizza (no DINO), DINO no disponible (abstención),
score bajo (abstención), disabled (legacy puro). Backward compat: disabled
= comportamiento legacy exacto (garantizado por la abstracción).

## Rollback
Verified: la config enabled=false devuelve legacy inmediatamente (test
unitario test_disabled_siempre_legacy ✓).

## Production Decision
APPROVE EXPERIMENTAL -> ROLLOUT CONTROLADO: legacy + DINO-base pizza/naan
(gate B, thr 0.40) reproduce 74.1% sin regresiones, reversible y con
latencia aceptable (+51ms promedio con el base). NO cambiar default
automáticamente: el rollout requiere shadow -> active con telemetría y
aprobación.

## Recommended rollout
1) Estado 1: legacy (actual).
2) Estado 2: shadow (resultado al usuario = legacy; telemetría del
   specialist en paralelo — medir correcciones/regresiones en tráfico real).
3) Estado 3: active (solo pizza/naan) con FOOD_AI_SPECIALIST_MODEL=dino_base,
   THRESHOLD=0.40, GROUPS=pizza_naan.
4) Rollback inmediato: enabled=false.

## F39
1) Siguiente confusion group CON EVIDENCIA INDIVIDUAL (el rice quedó
   demostrado negativo; buscar grupos donde el DINO-base aporte como en
   pizza/naan — la metodología F35/F37/F38 aplicada a candidatos nuevos).
2) Calibración de scores legacy vs DINO (temperature scaling si el base
   escala a más grupos).
3) Shadow mode en runtime real (tráfico, no solo benchmark) antes de
   activar el specialist en producción.
