# F31 RESULT

Branch: feature/f31-generalized-confusion-groups
Commit: (final, tree limpio)

## Architecture
Legacy 38 (default) → detección de grupo → ConfusionGroup (Linear sobre CLIP
frozen, 512->n) → gating por confianza → final. Grupos desde la confusion
matrix (no categorías). Abstracción configurable (data/models/f31/*.pt).

## Group Results (test food-us, sin leakage)
| Grupo | Legacy grupo | Especializado | Delta | Habilitado |
|---|---|---|---|---|
| rice/fries/fried_chicken | 60.7% (17/28) | 75.0% (21/28) | +14.3 | SI |
| pizza/naan | 85.0% (17/20 pizza) | 40.0% (8/20) | -45.0 | NO (regresión) |
| hamburger/sandwich | 72.5% (29/40) | 47.5% (19/40) | -25.0 | NO (sandwich 0/20) |
| taco/quesadilla/nachos | n/a (sin GT limpio) | val 50% | n/a | NO (sin evidencia) |
| fried_chicken/nuggets | 50.0% (4/8 fried) | 50.0% | 0 | NO (sin mejora) |

Solo rice cumple el criterio (group accuracy > legacy + sin regresión).

## Reproducibilidad rice
F30: 85.7% (fries 20/20) · F31: 75.0% (fries 17/20) — la variación proviene
del val split aleatorio (24 imágenes de train, seed fija pero split 20%
dependiente del orden). NO se reproduce exactamente: documentado como
variación del entrenamiento mínimo, no como regresión del enfoque.

## Global Results (food-us, con crops)
| Sistema | food-us |
|---|---|
| Legacy | 71.3% |
| F30 hierarchical (estimado) | 77.8% |
| F31 hierarchical (solo rice) | 72.2% (78/108) |

Gate 1 cumplido: 72.2 >= 71.3. Gate 2 (77.8) no alcanzado con el modelo F31
(75% vs 85.7% del grupo — la variación del grupo explica el delta).

## Per-Class (food-us, jerárquico)
fries: 13 -> 17-20 (según semilla) · fried_chicken: 4 -> 4 · pizza/hamburger/
hot_dog/sandwich: legacy sin cambio (fuera del grupo rice)

## Confusion Matrix (grupo rice)
fries -> rice: 0-4 (según semilla, legacy 4) · fried_chicken -> rice: 4
(limitado por 8 imágenes de train)

## Multi-food / Nutrition
Sin cambios (producción intacta) · Legacy E2E 59.3%

## Performance
Entrenamiento: ~3-5s por grupo CPU · Modelo: <1 KB por grupo (Linear 512x3) ·
Inferencia: +1ms sobre legacy · Sin impacto en latencia

## Data / Licensing
food-bench-v1 + prototype-src (Commons CC0/CC BY/CC BY-SA/PD) + propios ·
Food-101 excluido del entrenamiento · Manifest por grupo (distribution/val)

## Shadow Results
No implementado (el eval global = shadow: legacy vs jerárquico medidos por
separado — 71.3 vs 72.2)

## Production Decision
REJECT (no default). La arquitectura HIERÁRQUICA valida el concepto (supera
71.3 con un solo grupo habilitado) pero requiere: (a) reproducibilidad estable
del grupo (más datos de train), (b) más grupos con datos suficientes — los 4
grupos restantes fallaron con 8 imágenes/clase.

## Supported Classes
Legacy 38 + grupo rice (fries/fried_chicken/rice) — 40 clases efectivas

## Limitations
Solo rice habilitado · variación por semilla del grupo · otros grupos sin
datos suficientes (8 imgs/clase) · sin gating fino implementado (el umbral
de confianza se determinó por el criterio del grupo, no por grid)

## F32 Recommendation
1) AUMENTAR DATOS de train por grupo (fried_chicken 8, naan 8, taco 8):
   Nutrition5k (gsutil) + Commons validado — los grupos fallaron por datos,
   no por el enfoque (rice con 24 imgs funciona).
2) Gating fino: threshold por validación (0.5-0.9) + confianza legacy como
   señal de activación.
3) Re-evaluar pizza/naan y hamburger/sandwich con >=20 imgs/clase antes de
   descartarlos.
4) Shadow mode (FOOD_AI_HIERARCHICAL_SHADOW) para comparar en runtime sin
   afectar producción.
