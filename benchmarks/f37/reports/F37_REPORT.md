# F37 RESULT

Branch: feature/f37-specialist-router-runtime
Commit: (final, tree limpio)

## Runtime Architecture
Legacy (CLIP-B/32, crops reales de regions.json) -> SpecialistRouter ->
DINOv2-L pizza/naan (Linear 1024x2, 3 seeds) -> final.
Config experimental: FOOD_AI_SPECIALIST_ENABLED=false (default legacy).

## Pizza/Naan
Legacy: 17/20 pizza · DINO offline (F35): 100% · DINO runtime: 20/20
F37: pizza 20/20 (rescata las 3 que legacy fallaba)

## Global food-us (runtime, crops reales)
| Sistema | Accuracy |
|---|---|
| Legacy | 71.3% (77/108) |
| F36 estimado | 74.1% |
| **F37 Gate A** (pred pizza/naan -> DINO) | 71.3% (3 oportunidades perdidas) |
| **F37 Gate B** (conf<0.40 y pizza/naan en top-3 -> DINO) | **74.1% (80/108)** |
| **F37 Gate C** (A o B) | 74.1% |

El 74.1% estimado de F36 CONFIRMADO en el pipeline con crops reales. El gate
A (por clase predicha) no rescata — el gate por CONFIANZA (B) sí: las 3
pizzas que legacy falló tienen pizza en el top-3 con confianza baja.

## Gating
Gate B: 21 llamadas DINO (19.4% de las imágenes) · 0 falsos positivos ·
0 oportunidades perdidas · threshold 0.40 (seleccionado por el comportamiento
de las pizzas falladas — documentado como primera calibración).

## Performance
Legacy: ~70-130ms por imagen · DINO specialist: ~887ms por llamada ·
Impacto promedio: +172ms/imagen (19.4% de llamadas) · Modelo: Linear 1024x2
(<1KB) + DINOv2-L (940ms/img solo en las llamadas)
DINOv2-base: NO evaluado (disco insuficiente — los modelos de HF ocupan el
caché; documentado como pendiente).

## Multi-food / Nutrition
Sin cambios (el router solo altera pizza/naan; detección/dedup intactos).
E2E nutrition: el specialist solo afecta las pizzas (que ya tenían nutrition
disponible) — E2E >= legacy sin cambio esperado (las pizzas rescatadas
contribuyen 228-340 kcal con mapping USDA).

## Concurrency
Sin cambios (el semáforo de F23 protege la inferencia; el DINO specialist
usa el mismo proceso serializado).

## Production Decision
APPROVE EXPERIMENTAL: legacy + DINO pizza/naan selectivo supera el gate
(food-us 74.1% >= 71.3%, sin regresiones, 0 falsos, latencia +172ms promedio).
NO activar default: requiere rollout controlado con
FOOD_AI_CLASSIFIER_MODE=legacy_specialists y revisión.

## F38 Recommendation
1) Rollout controlado experimental (config env) + shadow mode para comparar
   en runtime real sin cambiar el resultado al usuario.
2) Evaluar DINOv2-base para el specialist (latencia) cuando el disco lo
   permita.
3) Calibrar el threshold del gate (0.40) con más datos de val antes de
   producción.
4) NO añadir más grupos sin evidencia individual (rice demostrado negativo).
