# Selección del modelo de profundidad (FASE 7)

Decisión documentada (2026-08-28). Criterio: calidad + CPU + tamaño + integración + licencia.

## Opciones evaluadas

| Opción | Pros | Contras |
|---|---|---|
| **Depth Anything V2 Small (elegido)** | Calidad top por tamaño; Apache-2.0; pipeline transformers 1-línea; ~90 MB; CPU viable (~0.8 s); fine-tuning futuro posible | Requiere transformers (~500 MB extra) |
| Depth Anything V2 Base/Large | Más precisión | 2-7× más pesado y lento; innecesario para evaluar señal |
| MiDaS (DPT) | Estable, MIT | Menos preciso que DA-V2, modelos más pesados |
| Depth Anything V3 | Más nuevo | Ecosistema/hf menos consolidado en el momento de elegir |

## Decisión: Depth Anything V2 Small (`depth-anything/Depth-Anything-V2-Small-hf`)

- **Versión**: `depth-anything-v2-small` (HF checkpoint, ~90 MB)
- **Licencia**: Apache-2.0 (repo oficial depth-anything) — compatible con el proyecto
- **Hardware**: CPU dev; 13 s de carga (una vez), ~0.76-0.86 s por imagen a resolución original
- **Limitación crítica**: la profundidad es **RELATIVA** (0..1), no una medición física. Sin referencia de tamaño conocido en la imagen (plato/cubierto) o calibración, NO es posible convertir a gramos.

## Configuración

```
FOOD_AI_PORTION_METHOD=basic|advanced
FOOD_AI_DEPTH_ENABLED=true
FOOD_AI_DEPTH_MODEL_PATH=depth-anything/Depth-Anything-V2-Small-hf
FOOD_AI_DEPTH_MODEL_VERSION=depth-anything-v2-small
FOOD_AI_DEPTH_DEVICE=cpu
```

## Cómo obtener el modelo

Se descarga automáticamente de Hugging Face al primer uso (cache local). Para
entornos sin red: descargar con `huggingface-cli download` y montar el cache
(Docker/AWS posteriormente).

## Conclusión experimental (FASE 7)

El depth map produce señal geométrica real (contraste alimento vs fondo:
banana 0.77, apple 0.75, pizza 0.12) pero **sin escala física no aporta
gramos**. Ver docs/portion-estimation.md para el veredicto completo.