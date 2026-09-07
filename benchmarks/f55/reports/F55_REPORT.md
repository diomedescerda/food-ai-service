# F55 RESULT — Active Rollout Controlado

Branch: feature/f55-active-rollout · Tests: 148/148 (6 nuevos) · Tree limpio

## Rollout (entorno controlado, food-us completo por API real)
- 108 requests con FOOD_AI_RETRIEVAL_ENABLED=true (respuesta = pipeline 5.761)
- ACTIVE top-1: 51/107 = 47.7% bruto — 51/72 = **70.8% sobre los alimentos
  con detección** (35 sin detección: foods vacío, igual que legacy)
- Legacy comparado: 47/107 = 43.9% bruto — 65.3% sobre los detectados
- El active supera al legacy (+5.5 pts sobre detectados; el delta vs shadow
  56.1% son los 9 casos donde el shadow evaluaba la imagen completa sin
  detección — el active no fabrica food sin bbox)

## Fix de integración
El active usaba el crop del clasificador del request (DetectorBased) en vez
del crop del CLIP del pipeline (padding 0.10) — 38.9 -> 47.7%. Ahora usa
pipeline.clf._crop (idéntico al shadow). Confidence del active: retrieval
score real (no 0.5 fijo).

## Fallback y rollback
- pipeline error -> legacy (DecisionPolicy pipeline_error) — probado
- FOOD_AI_RETRIEVAL_ENABLED=false -> legacy exacto (rollback inmediato por
  flag, sin redeploy; verificado OFF==ON en F54)
- specialist/DINO unavailable -> pipeline sin specialist (seguro)
- nutrition unavailable -> identificación intacta (no se vuelve unknown)

## Nutrition (estado actual)
2.184/5.761 (37.9%) — tandas reanudables independientes del rollout.

## Seguridad
0 crashes · 0 request failures · response contract intacto (mismo schema) ·
sin NaN · sin nutrientes inventados · sin detección -> foods vacío (no
fabricar comida).

## Performance
Active mode: ~2.5s/imagen p50 (pipeline 97ms + detector + porción);
nutrition lookup local (cero red). Sin trabajo innecesario (el shadow extra
no corre en active).

## Concurrencia
4 simultáneos: 200 x4, 0 crashes (verificado F54 con shadow; el active usa
el mismo semáforo).

## Producción (flags finales)
FOOD_AI_RETRIEVAL_ENABLED=false (default — el rollout se activa por
entorno) · NUTRITION_ENABLED=false · CONFIDENCE_ENABLED=false · shadows
false. El entorno controlado lo activa con el flag.

## Decisión
APPROVE F55 — READY FOR GRADUAL PRODUCTION: el pipeline activo responde
correctamente (70.8% sobre detectados, supera al legacy), fallback y
rollback probados, contract intacto, 0 errores. El rollout gradual (F56)
puede comenzar con el flag por entorno y legacy como red de seguridad.
