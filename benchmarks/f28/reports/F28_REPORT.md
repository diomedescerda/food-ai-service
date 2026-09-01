# F28 RESULT

Branch: feature/expand-visual-prototypes
Commit: (final de la rama, tree limpio)

## Prototype Coverage
- Total classes: 231
- Classes with prototypes: 33 (17 originales + 16 nuevas del food-bench-v1)
- Classes >=20: 0 · 10-19: 0 · 5-9: 30 · <5: 3 (banana/apple/orange 1)
- Classes without prototypes: 198 (INSUFFICIENT_DATA — sin imágenes permisivas disponibles)

## Prototype Quality
- Raw: 126 prototipos (F27) · Clean: 95 (VALID 72 + AMBIGUOUS 23)
- Rejected: 24 WRONG (ruido de título: salmon 5/8, steak 3, grilled_chicken 3, nuggets 3)
- Duplicates: 7 · Ambiguous: 23
- A/B calidad: raw vs clean IDENTICOS (food-us 60.2% = 60.2%) — la limpieza NO
  cambia el resultado (la agregación topk5/medoid diluye el ruido)

## Retrieval
- Recall@10: food-us 71.3% · v1 55.1% · Food-101 93.4% (sin cambio — retrieval textual)

## Classification (top1)
| Sistema | food-us | v1 | Food-101 |
|---|---|---|---|
| Legacy 38 | 71.3% | 47.0% | 89.4% |
| F26 (texto 231) | 34.3% | 26.3% | 49.6% |
| F27 (17 protos) | 60.2% | 37.2% | 74.9% |
| F28 (33 protos) | 59.3% | 55.1% | 85.0% |

## Nutrition
Legacy: E2E 59.3% food-us (producción intacta — no activada)
F27/F28: N/A (experimental, sin E2E)

## Multi-food
Precision 52.0% · Recall 56.5% · Duplicates 1 · Double-counting 2/2 (sin cambios)

## Performance
Latency p50: ~215ms (igual F27) · p95: ~280ms · Memory: prototipos 1373 KB
disco / ~0.5 MB RAM · Startup: +2-3s (embeddings 254×512)

## Per-Class (food-us — el regression set)
- pizza/hamburger/hot_dog: cubiertas con prototipos limpios (VALID) — el hybrid
  mantiene su nivel pero el texto legacy las clasifica mejor (gap ~1-3 pts/clase)
- french_fries/fried_chicken: la confusión → rice PERSISTE — los prototipos de
  fries no convencen al hybrid (w=0.25 el texto domina; w=0.5 baja el global)
- sandwich: sin nutrición (NO_RELIABLE) — sin cambio

## Main Errors
- fries→rice y fried_chicken→rice: estructural (visualmente similares); los
  prototipos NO la resuelven (el texto rice gana con w bajo)
- Prototype noise: irrelevante (A/B calidad lo demuestra)
- Missing prototype: las 198 clases sin imágenes — el v1 (+17.9 pts) demuestra
  que la COBERTURA es la variable que mejora

## Production Decision
231 classes: REJECT
Reason: F28 supera F26 y F27 en todos los datasets y SUPERA el legacy en el v1
(55.1 vs 47.0) y se acerca en Food-101 (85.0 vs 89.4), pero NO en el regression
set principal (food-us 59.3 vs 71.3 — el gap de fries/fried_chicken→rice).
El criterio del prompt (Nivel 2: alcanzar legacy sin regresiones) no se cumple
en food-us. La cobertura amplia es la dirección correcta (evidencia: +17.9 v1),
pero faltan las 198 clases (imágenes permisivas — Nutrition5k/Commons validado).

## Recommendation
F29: dos vías paralelas —
A) COBERTURA: prototipos para las 198 clases restantes (Nutrition5k CC BY 4.0
   vía gsutil + Commons con pipeline de validación F28); la evidencia del v1
   indica que la cobertura es lo que mueve la aguja.
B) CONFUSIÓN fries/fried_chicken→rice: experimento dirigido (prototipos
   específicos de rice vs fries vs fried_chicken + pesos por clase) antes de
   decidir fine-tuning (opción C del prompt).
Si F29 con cobertura total no cierra el gap de food-us: mantener legacy 38 en
producción (opción D) y documentar el techo del pipeline zero-shot.

## A/B respuestas (prompt §24-25)
- ¿La limpieza mejora? NO (raw = clean — evidencia incluida)
- ¿La cobertura resuelve? PARCIALMENTE: v1 supera legacy (+8.1) y Food-101 casi
  (94.8% del legacy); food-us no (el gap es de confusión visual, no de cobertura)
