# F36 RESULT

Branch: feature/f36-dino-hierarchical-classifier
Commit: (final, tree limpio)

## Dataset (validación RELAXED top-3, solo experimentación)
| Clase | STRICT | RELAXED | Total train |
|---|---|---|---|
| rice | 17 | 9 | 20 |
| french_fries | 16 | 3 | 12 |
| fried_chicken | 16 | 9 | 17 |
Total manifest_relaxed: 49 (35 strict + 14 relaxed) — el bloqueo de fries resuelto.

## Rice Group (test food-us 28, 3 seeds) — LA PREGUNTA DE F36
| Sistema | Accuracy |
|---|---|
| Legacy (completas) | 35.7% |
| F35 dino (fries 0) | 48.8% |
| F36 dino (fries 12 VALID) | 44.0% ± 22.7 |
| F36 clip_b32 (fries relaxed) | 15.5% ± 15.0 |
| Legacy (crops, ref) | 60.7% |

RESPUESTA: el 48.8% de F35 NO era (solo) datos — con fries VALID ampliadas
el DINO queda en 44%. El grupo rice es resistente al DINO frozen: la
separación de representación (inter 0.218) no se traduce en clasificación
sobre las imágenes reales del food-us. Las fries RELAXED además degradan el
clip_b32 (61.9 -> 15.5): el ruido de las relaxed perjudica al b32.

## Híbrido (Modelo C: legacy + DINO specialists)
pizza/naan dino = 100% (F35) -> legacy pizza 17/20 -> 20/20 (+3):
global food-us estimado = 80/108 = 74.1% (> legacy 71.3 ✓)
fried/nuggets: dino 83.3% = legacy -> sin aporte marginal.
rice: dino 44% < legacy 60.7 -> NO habilitar.

## 38 Classes
DINO global Linear (F35): 56.4% < legacy 71.3 — sin cambio (datos).

## Decision Tree aplicado
Caso E/C combinado: DINO mejora algunos grupos (pizza/naan 100%) pero NO
rice (44%) y su latencia es 7x. La arquitectura viable: legacy + DINO
specialist SOLO donde el DINO aporta (pizza/naan) — global estimado 74.1%.

## Production Decision
APPROVE EXPERIMENTAL: el híbrido legacy + DINO pizza/naan (74.1% estimado)
supera legacy; el rice requiere otra estrategia (fine-tuning de DINO o
análisis del GT — las imágenes fries del food-us pueden ser el límite).

## F37 Recommendation
1) Implementar el SpecialistRouter con DINO solo para pizza/naan (el grupo
   donde el DINO aporta 100% vs 88.3) — validar el global 74.1% en runtime.
2) rice: antes de más modelos, AUDITAR las 20 imágenes fries del food-us
   (¿el GT visual es separable? — el inter del dino 0.218 sugiere que el
   embedding las separa pero el test no).
3) DINOv2-base (menor latencia) para el specialist si el L no es viable.
