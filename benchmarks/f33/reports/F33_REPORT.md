# F33 RESULT

Branch: feature/f33-validated-data-selective-adaptation
Commit: (final, tree limpio)

## Data Quality (344 candidatas de los 5 grupos)
VALID 145 (42%) · AMBIGUOUS 58 · WRONG 41 · DUPLICATE 100
Manifest: data/training/f33/manifest.json (129 VALID tras dedup intersección)
Por clase (VALID): chicken_nuggets 32, fried_chicken 32, hamburger 32, pizza 32,
quesadilla 32, rice 32 · fries 0, naan 1 (tras dedup), sandwich 3, taco 0, nachos 0
Detalle: benchmarks/f33/reports/DATA_QUALITY.md

## Deterministic Splits
Hash md5(path) % 20 (train 0-13 / val 14-16) · test externo food-us (sin leakage)

## Group Results (test food-us, multi-seed, datos VALID)
| Grupo | Legacy grupo | Frozen (VALID) | LoRA | Status |
|---|---|---|---|---|
| pizza/naan | 85.0% | 96.7% ± 4.7 | — | HABILITABLE |
| fried_chicken/nuggets | 50.0% | 100.0% ± 0.0 | 50.0% | HABILITABLE |
| rice/fries/fried_chicken | 60.7% | 29.8% ± 21.9 | — | degradado (validación excluyó las fries difíciles) |
| hamburger/sandwich | 72.5% | no entrenado (sandwich 3 VALID) | — | INSUFFICIENT_VALID_DATA |
| taco/quesadilla/nachos | n/a | sin GT limpio | — | INSUFFICIENT_EVIDENCE |

LoRA: 50% < frozen 100% -> NO conservado (criterio: LoRA > frozen)

## Gating
Flujo por predicción legacy (prompt §8): el especializado solo ve las imágenes
que legacy ya predijo en el grupo -> CONFIRMA las correctas, no rescata las
falladas. Global con pizza/naan + fried/nuggets (gating 0.0-0.6): 71.3% = legacy.

## Global
| Sistema | food-us |
|---|---|
| Legacy | 71.3% |
| F31 (rice limpio) | 72.2% |
| F33 (pizza/naan + fried/nuggets) | 71.3% (= legacy) |

## Key Finding
La validación F33 arregló fried/nuggets (33±47 crudo -> 100±0 VALID) y
pizza/naan (86.7 -> 96.7) PERO el flujo por predicción-legacy no los traduce
al global (solo confirma). El rice (el único que movía el global en F31)
degradó con la validación estricta (las fries difíciles eliminadas).

## Production Decision
REJECT — F33 demuestra que la CALIDAD de datos es la palanca correcta (los
grupos validados superan legacy en su grupo) pero el GATING por predicción
legacy es el cuello: el especializado no puede rescatar las imágenes que el
legacy ya clasificó mal. El gating alternativo (legacy-confidence bajo ->
especializado, prompt §25) es el siguiente paso.

## F34 Recommendation
1) Gating por legacy-confidence: activar el especializado cuando la confianza
   del legacy es BAJA (las imágenes dudosas) — el flujo actual solo confirma.
2) rice: relajar la validación para fries (top-3 en vez de top-1) para
   recuperar el grupo que demostró mover el global (F31 72.2).
3) hamburger/sandwich: conseguir >8 sandwich VALID (el inter 0.403 sugiere
   que el embedding las separa).
