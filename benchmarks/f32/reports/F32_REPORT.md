# F32 RESULT

Branch: feature/f32-data-gating-stability
Commit: (final, tree limpio)

## Data
| Group | Imágenes/clase | Train | Val | Test_int |
|---|---|---|---|---|
| rice/fries/fried_chicken | 27 | 65 | 16 | 15 |
| pizza/naan | 23-25 | 38 | 10 | 8 |
| hamburger/sandwich | 27-29 | 48 | 8 | 8 |
| taco/quesadilla/nachos | 24-28 | 71 | 7 | 18 |
| fried_chicken/nuggets | 27-29 | 47 | 9 | 8 |
Fuentes: Commons (24 nuevas/clase, CRUDAS sin pipeline de validación) + food-bench
+ prototype-src previos. Test externo: food-us (sin leakage — hash split por path).

## Split Reproducibility
- Hash: md5(path) % 20 (train 0-13 / val 14-16 / test 17-19) — determinista
- Leakage: test externo food-us nunca en train (datasets distintos) ✓
- Seeds: 42/123/2026 — mean ± std reportados

## Group Results (test externo food-us, multi-seed)
| Grupo | Legacy | F31 | F32 mean ± std | Clasificación |
|---|---|---|---|---|
| rice/fries/fried_chicken | 60.7% | 75.0% | 46.4% ± 0.0 (fries 13/20, fried 0/8) | DATA-QUALITY-LIMITED |
| pizza/naan | 85.0% (pizza) | 40.0% | 86.7% ± 8.5 (pizza 15/20) | DATA-LIMITED (datos ayudan) |
| hamburger/sandwich | 72.5% | 47.5% | 50.0% ± 0.0 (hamburger 20/20, sandwich 0/20) | TRAIN-QUALITY-LIMITED |
| taco/quesadilla/nachos | n/a | n/a | sin GT limpio | DATA/TEST-LIMITED |
| fried_chicken/nuggets | 50.0% | 50.0% | 33.3% ± 47.1 (inestable) | REPRESENTATION-LIMITED |

## Data Scaling (el experimento clave)
rice: 8 imgs/clase (F31) = 75% · 27 imgs/clase CRUDAS (F32) = 46.4% — ¡MÁS
DATOS CRUDOS DEGRADAN! El punto de saturación de rice es ~8-12 imágenes LIMPIAS
(el pipeline de validación F28 es obligatorio antes de ampliar).
pizza/naan: 8 = 40% → 24 = 86.7% — los datos SÍ ayudan aquí.
hamburger/sandwich: 8 → 24: sandwich sigue 0/20 — los datos no resuelven.

## Gating
Threshold 0.6 (softmax del especializado) aplicado en el global. El gating
protege (el global no cae por debajo de legacy con rice+pizza_naan) pero no
aporta mejora con los modelos crudos.

## Global
| Sistema | food-us |
|---|---|
| Legacy | 71.3% |
| F31 (rice limpio 8/clase) | 72.2% |
| F32 rice+pizza_naan (crudo) | 71.3% |
| F32 +hamburger_sandwich | 59.3% |

F32 no supera F31 — la ampliación sin validación de calidad degradó el grupo
que funcionaba.

## Representation Diagnostics (CLIP embeddings)
- rice/fries/fried: intra 0.645-0.726 · inter 0.563-0.615 (gap 0.08-0.13) —
  clases cercanas; la frontera es aprendible con datos LIMPIOS (F31 75% lo
  demostró) pero frágil
- pizza/naan: intra 0.747/0.748 · inter 0.612 (gap 0.13) — separable ✓
- hamburger/sandwich: intra 0.69/0.73 · inter 0.403 (gap 0.29!) — MUY separable;
  el sandwich 0/20 NO es representación — es la calidad del train
- fried/nuggets: intra 0.695/0.686 · inter 0.637 (gap 0.05) — NO separable
  (representación limitada -> fine-tuning/LoRA en F33)

## Multi-food / Nutrition
Producción intacta · Legacy E2E 59.3%

## Performance
Entrenamiento ~5s por grupo/seed · modelos <1 KB · gating +1ms

## Production Decision
REJECT — F32 demuestra que CANTIDAD sin CALIDAD degrada (rice 75 -> 46.4) y
que la arquitectura jerárquica con datos limpios (F31) sigue siendo el mejor
resultado (72.2%). El global con datos crudos = legacy (71.3%).

## F33 Recommendation
1) APLICAR el pipeline de validación F28 (VALID/AMBIGUOUS/WRONG) a las 24
   imágenes nuevas por clase antes de entrenar — el ruido de Commons es la
   causa de la degradación del rice.
2) Re-evaluar con ~12 imágenes VALIDADAS por clase (el punto de saturación
   de rice).
3) LoRA/adaptación de CLIP para fried/nuggets y rice (representación
   limitada: inter ≈ intra).
4) Re-entrenar sandwich con imágenes validadas (el inter bajo sugiere que el
   problema es el train, no el embedding).
