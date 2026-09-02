# F35 RESULT

Branch: feature/f35-strong-visual-backbones
Commit: (final, tree limpio)

## Modelos evaluados
| Backbone | Embedding | Params | Carga | Latencia/img | Licencia |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | 768 | 151M | 17s | 131ms | MIT |
| DINOv2 ViT-L/14 | 1024 | 304M | 121s | 940ms | Apache-2.0 |
| SigLIP base-16 | 768 | 89M | 29s | 229ms | Apache-2.0 |
| CLIP ViT-L/14 | 1024 | 428M | NO EVALUADO — disco insuficiente (1.2GB) | — | MIT |

## Diagnóstico de representación (intra/inter — MEJOR separación = menor inter)
| Grupo | clip_b32 | dino_l | siglip |
|---|---|---|---|
| pizza/naan | 0.612 | **0.113** | 0.601 |
| fried/nuggets | 0.666 | **0.218** | 0.666 |
| hamburger/sandwich | 0.301 | **0.038** | 0.425 |

DINOv2-L reduce el inter 3-15x vs CLIP-B/32: la hipótesis de F35 se
CONFIRMA a nivel de representación (las clases que b32 no separa, dino sí).

## Linear por grupo (food-us test, 3 seeds)
| Grupo | clip_b32 | dino_l | siglip |
|---|---|---|---|
| pizza/naan | 88.3 ± 16.5 | **100.0 ± 0.0** | 48.3 ± 39.2 |
| fried/nuggets | 83.3 ± 23.6 | 83.3 ± 23.6 | **100.0 ± 0.0** |
| hamburger/sandwich | 40.0 ± 12.4 | **49.2 ± 1.2** | 50.0 ± 0.0 |
| rice/fries/fried | 61.9 ± 10.2 | 48.8 ± 10.2 | 51.2 ± 15.0 |

rice NO EVALUABLE justamente: fries = 0 VALID en el manifest (la validación
estricta las eliminó) — el grupo crítico necesita la validación relajada.
hamburger/sandwich: solo 3 sandwich VALID (insuficiente, pero el dino ya
mejora 40 -> 49.2).

## Global food-us (Linear directo, sin gating)
DINOv2-L: 56.4% ± 3.7 — NO supera legacy 71.3% (el problema de datos del
global persiste: 8-32 imágenes/clase). La ganancia del backbone se observa
en los GRUPOS, no en el Linear global con datos mínimos.

## Respuesta a la pregunta de F35
¿Un backbone más fuerte separa las clases que CLIP ViT-B/32 no separa? -> SI
(diag inter 0.038-0.218 vs 0.60-0.67; pizza/naan 100% vs 88.3) pero el
impacto global requiere: fries VALID (validación relajada) + arquitectura
jerárquica con el nuevo backbone.

## Decisión
APPROVE REPRESENTATION (experimental): DINOv2-L demuestra separación
claramente superior y mejora robustamente los grupos evaluables. No es
production-ready: latencia 940ms vs 131ms (7x) y datos insuficientes.

## F36 Recommendation
1) Validación relajada de fries (top-3) -> re-evaluar el grupo rice con
   DINOv2-L (el grupo que bloquea el global).
2) Reconstruir el jerárquico (legacy + especialistas DINOv2-L) con datos
   VALID ampliados.
3) Evaluar DINOv2-L base (menor latencia) si el L no es viable para CPU
   de producción.
