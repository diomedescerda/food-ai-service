# F34 RESULT

Branch: feature/f34-confidence-gating-rescue
Commit: (final, tree limpio)

## Baselines
Legacy val: 69.0% (29) · Legacy test: 72.2% (79) · Legacy completo: 71.3% (108)
F33: 71.3% (flujo por predicción)

## Gating implementado
Legacy top-1 + confidence -> si conf < T -> especialistas de las clases del
TOP-3 del legacy -> regla S/M -> final. El especialista NO está restringido a
la predicción legacy (corrige el defecto de F33).

## Threshold Sweep (validation, S=0 M=0 — potencial máximo)
| T | gated | acc | rescue | harmful | net |
|---|---|---|---|---|---|
| 0.20 | 0% | 69.0% | 0 | 0 | 0 |
| 0.25 | 21% | 65.5% | 0 | 1 | -1 |
| 0.30 | 66% | 55.2% | 0 | 4 | -4 |
| 0.35+ | 100% | 44.8% | 0 | 7 | -7 |

RESCUE = 0 EN TODOS LOS THRESHOLDS. El especialista NUNCA corrige un error
del legacy: los errores están CORRELACIONADOS (las imágenes difíciles para el
legacy lo son para el especialista — el CLIP embedding no las separa).

El routing por top-3 además activa grupos IRRELEVANTES (pizza en el top-3 de
una hamburguesa -> el especialista pizza/naan la "corrige" a pizza ->
harmful).

## Sweep S/M (validation)
S=0.0-0.5, M=0.05-0.2: acc constante 69.0% (net=0) — el S alto rechaza todo
el gating (sin cambios); el S bajo solo añade harmful.

## Resultado Final (test)
T=0.2 (sin gating) = 72.2% = legacy. El confidence gating NO aporta.

## Respuesta a la pregunta de F34
NO: el gating por confianza no permite que los especialistas (96.7% y 100% en
su grupo) corrijan errores del legacy sin introducir más errores de los que
corrigen (rescata 0, daña hasta 7). Los errores de clasificación del CLIP son
consistentes entre el clasificador general y los especialistas entrenados
sobre el mismo embedding.

## Production Decision
REJECT — y CIERRE DEL PARADIGMA frozen-CLIP + confusion groups según el
criterio del prompt ("si falla incluso con especialistas de 96.7% y 100%").
El techo del sistema: legacy 71.3-72.2% food-us.

## F35 Recommendation
Evaluar un backbone visual más fuerte (DINOv2 ViT-L / SigLIP / CLIP-L) como
REPRESENTACIÓN base del clasificador aprendido — la evidencia acumulada
(F26-F34) muestra que el embedding CLIP ViT-B/32 es el límite: no separa las
clases difíciles (fries/rice/fried_chicken) ni permite a clasificadores
entrenados rescatar sus errores. Alternativa complementaria: clasificador
entrenado con más datos por clase (el Linear global falló con 8 imgs/clase).
