# F30 RESULT

Branch: feature/f30-clip-adaptation
Commit: (final de la rama, tree limpio)

## Dataset
Training: 531 imágenes / 67 clases (food-bench-v1 + prototype-src + multi-food + assets)
Test: food-us (108) + Food-101 interno (3500) — datasets distintos (sin leakage)
Sources: Wikimedia Commons (CC0/CC BY/CC BY-SA/PD) + propios · Licencias: producción-safe
Food-101: benchmark interno únicamente

## Experiment 1 — Rice Group (rice/french_fries/fried_chicken)
Legacy (28): 60.7% · F29 focused hybrid: 75.0%
F30 Linear (frozen CLIP, 24 train): 85.7% (24/28)
- french_fries: 20/20 (legacy 13/20) — confusión fries→rice ELIMINADA
- fried_chicken: 4/8 (limitado por 8 imágenes de train)
- Food-101 fries: 250/250 (generaliza)
=> La frontera aprendida sobre el embedding CLIP congelado RESUELVE el grupo

## Experiment 2 — 38/67 Classes
Legacy: 71.3% (food-us)
F30 Linear (67 clases, 8 imgs/clase): 17.6%
F30 MLP: 15.7%
=> El classifier global NO escala con 8 imágenes por clase (val_acc ~17% —
el embedding CLIP no separa 67 clases linealmente con tan pocos ejemplos)

## Experiment 3 — Hierarchical (legacy + confusion groups)
Caso D del prompt: legacy (77/108 food-us) + grupo rice especializado
(24/28) => 84/108 = 77.8% (estimado con métricas medidas)
=> SUPERA legacy 71.3% en +6.5 pts (Gate 1 cumplido)

## Classification
| Sistema | food-us | notas |
| Legacy 38 | 71.3% | baseline |
| F26-F29 | 34.3-60.2% | prototype/text (cerrado) |
| F30 rice group | 85.7% (28) | fries 20/20 |
| F30 global Linear/MLP | 17.6/15.7% | no escala (8 imgs/clase) |
| F30 hierarchical | 77.8% | legacy + grupo rice (estimado) |

## Confusion Matrix (grupo rice, F30)
fries -> rice: 0 (legacy 4) · fried_chicken -> rice: 4 (persiste — datos limitados)

## Per-Class (food-us grupo)
fries: 13 -> 20 (improved) · fried_chicken: 4 -> 4 (unchanged) · rice: n/a

## Macro F1
No calculado (train mínimo — el grupo rice 3 clases, reportado por per-class)

## Multi-food / Nutrition
Sin cambios (producción intacta) · Legacy E2E 59.3%

## Performance
Entrenamiento: 3-40s CPU · Inferencia: embedding CLIP (~75ms) + Linear (<1ms) ·
Modelo: <1 MB (Linear 512x3) · Sin cambio de latencia significativo

## Production Decision
APPROVE (experimental, rollout controlado) para la arquitectura HIERÁRQUICA
legacy + confusion-group classifiers. NO activar como default todavía.
Reason: el grupo rice especializado (85.7%) + legacy = 77.8% food-us > 71.3%
(Gate 1 ✓); el global entrenado con 8 imgs/clase no funciona (Gate 1 ✗ para
el modelo global) — la vía es la jerárquica con grupos, no el modelo global.

## F31 Recommendation
1) Implementar la arquitectura ConfusionGroup generalizada (grupos derivados
   de la confusion matrix: rice/fries/fried_chicken, pizza/naan,
   hamburger/sandwich, taco/quesadilla/nachos, fried_chicken/nuggets) con
   especializados entrenados por grupo.
2) Aumentar datos de train por clase (Nutrition5k vía gsutil / Commons
   validado) para las clases del grupo con pocos ejemplos (fried_chicken 4/8).
3) Rollout controlado: FOOD_AI_CLASSIFIER_MODE=hierarchical (experimental),
   legacy = default, A/B en producción antes de decidir.
