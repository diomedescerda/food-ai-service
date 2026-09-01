# F29 RESULT

Branch: feature/f29-prototype-scale-rice-resolution
Commit: (final de la rama, tree limpio)

## Coverage
Classes: 68 (33 previas + 36 nuevas de Commons prototype-src; wrap 0/8 sin datos)
Prototypes: 534 (2892 KB)
Classes without prototypes: 163 (INSUFFICIENT_DATA)

## Rice Experiment (food-us, 28 imágenes fries+fried_chicken)
Baseline legacy: 17/28 = 60.7% (fries 13 + fried 4)
Hybrid enfocado w=0.5 (solo rice/fries/fried_chicken): 21/28 = 75.0% (+14.3)
Errores restantes: 4 -> rice, 3 -> fried_chicken (de 7)
- La confusión rice SE REDUCE con scoring enfocado pero no desaparece
- Causa: CLIP no distingue plenamente las 3 clases (texto y visual ambos
  favorecen la clase equivocada en 7 casos)

## Coverage Experiment (food-us top1)
33 clases: 59.3% · 68 clases: 38.9% (TOP_N=10) / 38.0% (TOP_N=30) -> DEGRADA
v1: 33->55.1% · 68->53.4%
Food-101: no re-medido en 68 (la tendencia de food-us/v1 es suficiente)
CONCLUSION: la cobertura amplia NO escala con el scoring actual — las clases
nuevas (prototipos ruidosos de Commons) compiten y degradan el regression set

## Classification (top1)
| | food-us | v1 |
| Legacy | 71.3% | 47.0% |
| F26 | 34.3% | 26.3% |
| F27 (17) | 60.2% | 37.2% |
| F28 (33) | 59.3% | 55.1% |
| F29 (68) | 38.9% | 53.4% |

## Per-Class
- rice/fries/fried_chicken: 75% con scoring enfocado (el enfoque dirigido
  funciona en el grupo específico)
- Clases nuevas (naan/tikka/butter_chicken/curry...): sus prototipos roban
  fotos del food-us (pizza/hamburguesa) -> degradación global

## Confusion Pairs (food-us)
fries -> rice (4) · fried_chicken -> rice (4 legacy) · fried_chicken ->
fried_chicken WRONG direction (3) — grupo rice confirmado como principal

## Multi-food / Nutrition
Sin cambios (producción intacta) · Legacy E2E 59.3% vigente

## Performance
68 clases: ~215-280ms (el matmul de 534 prototipos sigue trivial) · 2.9 MB store

## Licensing
prototype-src: Wikimedia Commons CC0/CC BY/CC BY-SA/PD (metadata por imagen)
Food-101: EXCLUIDO · Nutrition5k: no descargado (gsutil ausente — limitación)

## Production Decision
231 classes: REJECT
Why: la cobertura amplia DEGRADA el regression set (59.3 -> 38.9 al pasar de
33 a 68 clases) — el scoring hybrid no escala; el techo del prototype approach
con CLIP (texto+visual) en food-us es ~60% vs legacy 71.3%. El caso del prompt
§25 se confirmó: la cobertura no es la palanca (degradación con ruido).

## F30 Recommendation
DETENER la expansión masiva de prototipos. El benchmark demuestra un techo
arquitectónico del scoring actual. F30 debe elegir entre:
A) fine-tuning/adaptación de CLIP (el gap de 11 pts requiere representación
   mejor, no más datos — evidencia: 75% con scoring enfocado sugiere que un
   clasificador específico del grupo rice puede funcionar)
B) clasificador entrenado ligero por grupos de confusión (rice/pizza-naan/...)
C) mantener legacy 38 en producción (opción D del prompt F29)
La evidencia de F29 favorece A/B sobre más prototipos.
