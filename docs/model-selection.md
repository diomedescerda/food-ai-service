# Selección del modelo de detección

Decisión documentada (2026-08-27). Criterio MVP: integración + calidad razonable + velocidad + fine-tuning futuro + licencia.

## Opciones evaluadas

| Opción | Pros | Contras |
|---|---|---|
| **Ultralytics YOLO11n (elegido)** | Integración 1-línea (`YOLO(...)`), CPU ok, fine-tuning directo con el mismo ecosistema, docs extensas, comunidad grande | Licencia AGPL-3.0; solo 10 clases de comida en COCO; depende de PyTorch (imagen grande) |
| Ultralytics YOLOv8n | Mismo ecosistema, más maduro | Mismo AGPL; ligeramente más pesado |
| YOLO-World | Open-vocabulary (detecta "arroz" sin dataset) | Más pesado, fine-tuning menos directo, ecosistema más nuevo |
| RT-DETR | Arquitectura transformer, buena precisión | Más lento en CPU, integración menos directa |
| Modelos food-especializados (FoodDet-Benchmark, MangoNet, etc.) | Clases de comida reales | Mal mantenidos, sin ecosistema, difíciles de desplegar/fine-tunear |
| Grounding DINO | Open-vocabulary robusto | Muy pesado, lento CPU, fine-tuning complejo |

## Decisión: YOLO11n (Ultralytics) sobre COCO

- **Versión**: `yolo11n.pt` (v8.3.0 assets) — modelo nano, ~5.6 MB
- **Licencia**: AGPL-3.0 (Ultralytics). Aceptable para desarrollo interno; **revisar antes de distribución comercial** (o migrar a YOLO con licencia permisiva/empresarial).
- **Clases de comida soportadas (COCO)**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake (10 de 80). `bowl` se excluye (vajilla, no alimento).
- **Hardware**: CPU sin GPU (dev). ~50–200 ms por imagen en CPU moderna a 640 px.
- **Fine-tuning futuro**: mismo formato de anotación YOLO → dataset colombiano (FASE 10–12) directo.
- **Model versioning**: `food-detector-v1` (settings `FOOD_AI_MODEL_VERSION`). Nunca se sobrescribe un modelo; cada análisis reporta su versión.

## Threshold

`FOOD_AI_CONFIDENCE_THRESHOLD = 0.35` inicial (default Ultralytics). Configurable; experimentar 0.25–0.60 midiendo impacto (FASE 19).

## Estrategia de despliegue del modelo

- Local: `weights/yolo11n.pt` (gitignoreado; descarga documentada abajo).
- Docker: el modelo se copia en **build** (cache de capas). En producción se montará/descargará de almacenamiento externo (S3) con versionado (FASE 20).

## Cómo obtener el modelo

```bash
mkdir weights
curl -L -o weights/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

## Limitaciones conocidas

- Clases limitadas a las 10 de comida de COCO (no arroz/arepa/frijoles — fase de dataset colombiano lo resuelve).
- Detecta objetos individuales; platos compuestos (bandeja paisa) = múltiples cajas, sin descomposición semántica (FASE 4/9).
- Sin estimación de porción ni nutrición (fases 5–8).
- AGPL-3.0: revisar para producción comercial.