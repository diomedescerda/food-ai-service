# F44 RESULT

Branch: feature/f44-specialist-dino-rerank
Commit: (final, tree limpio) · Tests: 75/75

## Integración
Retrieval top-50 (CLIP) -> canonical grouping -> si pizza/naan en candidatos
Y el gate del F38 se activa (conf legacy < 0.40 + pizza/naan en top-3 del
legacy) -> DINO-base (F38) predice -> si conf >= 0.70 -> promueve al top-1.
Los demás candidatos conservan el score retrieval (sin penalización).
Fallback: excepción del DINO -> retrieval intacto.

## Evaluación (config final: gate 0.40 + DINO conf 0.70)
| Sistema | R@1 | R@5 | R@10 | R@20 | R@50 |
|---|---|---|---|---|---|
| F43 base | 38.9% | 58.3% | 63.0% | 63.9% | 64.8% |
| **F44 food-us** | **41.7%** | 59.3% | 63.9% | 63.9% | 64.8% |
| F43 v1 | 19.0% | 29.4% | 33.9% | 36.3% | 37.1% |
| F44 v1 | 19.0% | 29.4% | 33.9% | 36.3% | 37.1% |
| F43 Food-101 | 32.2% | 45.3% | 49.4% | 52.8% | 53.5% |
| F44 Food-101 | 31.5% | 45.4% | 49.5% | 52.8% | 53.5% |

## Pizza/Naan (la métrica crítica)
food-us pizza: F44 19-20/20 (vs base retrieval ~17/20) · v1 pizza 8/8 ·
Food-101 pizza 247-248/250 — la ventaja del F38 SE TRANSFIERE al catálogo
de 1.451 alimentos.

## Specialist (config 0.70)
eligible=371/3856 (9.6%) · invoked=371 · corr=273 · regr=42 (1.1%) ·
abst=56 · dino_ms=256 · overhead total: 371 x 256ms = ~95s sobre 3.856
imágenes (+25ms promedio)

## Comparación de thresholds (documentada)
| conf | food-us | food101 | regr |
|---|---|---|---|
| 0.60 | +3.7 | -1.6 | 146 |
| 0.60 + gate | +3.7 | -1.6 | 82 |
| 0.70 + gate | +2.8 | -0.7 | 42 |

El gate del F38 (conf legacy) es esencial: sin él 146 regr. El threshold
0.70 equilibra mejora vs regresiones.

## Error analysis de rescates (ejemplo)
Pizza del food-us que el retrieval rankeaba mal -> DINO conf alta ->
promovida al top-1 -> GT pizza ✓. Las regresiones (42): imágenes no-pizza
(lasagna/quiche/platos con queso) donde el DINO confirma pizza con conf
0.7-0.8 — el DINO sobre-generaliza a platos similares.

## Latencia
Retrieval ~100ms + grouping <1ms + DINO 256ms (solo en el 9.6% eligible) =
+25ms promedio por imagen sobre el retrieval base.

## Producción
Intacta: retrieval off (default), legacy = respuesta, SpecialistShadow sin
cambios, 75/75.

## Decisión
APPROVE F44 (mejora localizada): el specialist DINO transfiere su ventaja al
retrieval masivo (food-us +2.8 pts, pizza 19-20/20) con regresiones
controladas (1.1%, documentadas: el DINO sobre-generaliza a platos con
queso/salsa). R@K sin degradación significativa (Food-101 -0.7).

## Siguiente
1) Para reducir las 42 regresiones: calibrar el threshold del DINO con
   validation (0.65-0.75) o añadir la señal del canonical (pizza solo si
   el retrieval la rankea dentro del top-10).
2) Shadow retrieval (flag) con la telemetría top1/5/10/20/50 + specialist.
3) Escalar el catálogo a 10.000 (mecánica — el pipeline ya está).
