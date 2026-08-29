# Clasificación de alimentos (FASE 4)

## Decisión: clasificación del detector es suficiente — sin segundo modelo

YOLO11n/11n-seg (Ultralytics) es un detector **+ clasificador integrado**: cada
detección ya incluye clase y confidence. Para las 10 clases de comida de COCO
(pizza, banana, apple, ...) un clasificador separado (p. ej. sobre crops) no
aportaría precisión medible y añadiría latencia, memoria y un modelo más que
mantener. **No se agregó un segundo modelo.**

## Detección vs Clasificación

| Etapa | Pregunta | Fuente actual |
|---|---|---|
| Detection | ¿Dónde hay un objeto? ¿Qué clase parece? ¿Confianza? | YOLO11n |
| Segmentation | ¿Cuál es la forma exacta del objeto? | YOLO11n-seg |
| Classification | ¿Qué alimento es EXACTAMENTE este objeto? | clase+confidence de YOLO (`DetectorBasedClassifier`) |

Cuando el catálogo crezca a comida colombiana (arroz blanco vs arroz con coco,
arepa, patacón, frijoles...), YOLO-COCO no distinguirá: ahí entra un
clasificador dedicado.

## Arquitectura

```
IFoodClassifier (app/models/classifier_base.py)
    │
    └── DetectorBasedClassifier (app/models/detector_based_classifier.py)
            └── IFoodDetector (delega: clase+confidence, SIN inferencia extra)
```

- El pipeline (`analyze.py`) trabaja contra `IFoodClassifier` — nunca contra un
  modelo concreto.
- `DetectorBasedClassifier.classify()` es O(1) por detección: no llama al
  modelo. El costo de clasificación es 0 ms.
- Para el futuro: `ColombianFoodClassifier` implementa `IFoodClassifier`
  (fine-tuning de cropped masks/bboxes) y se reemplaza en `main.py` sin tocar
  `analyze.py`.

## Contrato

`/analyze` añade `classifier_version` (opcional, retrocompatible):

```json
{
  "analysis_id": "uuid",
  "status": "completed",
  "model_version": "food-detector-v1",
  "seg_model_version": "food-segmenter-v1",
  "classifier_version": "detector-based-v1",
  "inference_time_ms": 5185,
  "foods": [{ "name": "pizza", "confidence": 0.9253, "bounding_box": {...}, "segmentation": {...} }]
}
```

El `name`/`confidence` finales del alimento provienen de la clasificación
(actualmente idénticos a la detección).

## Evaluación real (imágenes de prueba)

| Imagen | Clase | Confidence |
|---|---|---|
| pizza.jpg | pizza | 0.9253 |
| banana.jpg | banana | 0.8932 |
| apple.jpg | apple | 0.8613 |

Todas coherentes con las clases soportadas; confianza > threshold 0.35.

## Limitaciones

- 10 clases COCO: sin comida colombiana (arroz, arepa, frijoles, sancocho...).
- Platos compuestos (bandeja paisa) → múltiples detecciones sin relación
  semántica entre ellas (requiere LLM auxiliar — FASE 9).
- `DetectorBasedClassifier` no distingue variedades (arroz blanco vs coco).
- Evaluación formal (accuracy por clase, matriz de confusión) → FASE 19.

## Qué será necesario para comida colombiana

1. Dataset de 30-50 clases (FASE 10-11).
2. `ColombianFoodClassifier` con fine-tuning de YOLO sobre crops/masks (FASE 12).
3. Sustituir `DetectorBasedClassifier` en DI/lifespan — sin cambios en analyze.
## CLIP zero-shot (FASE 10B)

- Modelo: openai/clip-vit-base-patch32 (MIT), version clip-zero-shot-v1, CPU, carga única (~13 s)
- CLIP clasifica CROPS de las detecciones de YOLO (bbox + padding configurable), no la imagen completa (el embedding global pierde alimentos en multi-food — verificado en benchmark).
- Catálogo: 39 entradas en app/models/food_catalog.py (canonical_name + clip_candidates + category + nutrition_key). Agregar alimento = agregar entrada (sin reentrenar).
- Prompt: 'a photo of {food}' (configurable). Features de texto precomputadas en load.
- Threshold: clip_threshold=0.22 (scores por crop típicos 0.2-0.45; por debajo → UNKNOWN, nunca se fuerza clase).
- A/B: FOOD_AI_CLASSIFIER_TYPE=detector_based|zero_shot (config, sin código).
- Esto es zero-shot CLASSIFICATION (matching texto-imagen contra catálogo fijo); NO es vector search/RAG (sin BD vectorial).

## Tuning FASE 14 — experimentos controlados (resultados medidos)

- Baseline (prompt 'a photo of {food}', padding 0.05, thr 0.22): Top-1 0.648
- GANADOR: ensemble mean (a photo/a picture/a close-up photo) + padding 0.10 + thr 0.20 → Top-1 0.685 (+0.037)
- E2E Nutrition: 28.7% → 30.6% (pizza 16→18)
- Fallidos: prompts específicos (0.426), jerárquico (0.352), masked crop (=baseline), padding 0.0 (0.593)
- Threshold: 0.20 (balance accuracy-classified 0.72 / unknown 0.20) vs 0.22 (0.71 / 0.36)
- Confusiones dominantes: fries→rice (7), fried_chicken→rice (4), pizza→lasagna (3), sandwich→grilled_chicken (3)
- Top-K: solo 31.6% de los fallos top-1 tienen el correcto en top-3 → problema de representación/prompts, no de re-rank
- Trade-off: el ensemble degradó hot_dog (13/20 rank simple → fuera de top-1 con ensemble) — aceptado por la ganancia global
- Config: FOOD_AI_CLIP_PROMPT_ENSEMBLE='a picture of {food}|a close-up photo of {food}'

## FASE 16 — scoring por clase

El ranking global de candidatos perdía clases con scores medios (hot_dog: 0/20 con ensemble global). FIX: score(clase) = max de los candidatos de la clase → hot_dog 13/20 sin degradar el global (68.5%). Implementado en ZeroShotFoodClassifier (score_by_class=True).

## FASE 17 — experimentos de representación y multi-instancia

- **Crop (fries/fried_chicken, regiones hybrid)**: padding 0.0→fries 6/20, 0.10→7/20, 0.20→7/20, masked 0.10→3/20 (PEOR). El crop no es el cuello de botella: la confusión fries→rice es estructural (CLIP ve masa granulada dorada). Se mantiene padding 0.10; masked descartado de nuevo.
- **Candidate groups específicos** (fries/fried_chicken): candidatos descriptivos extra → top1 sin cambio (39.3%→39.3%): fries→rice bajó 7→4 pero fried_chicken subió 3→6. NET ZERO → rechazado (regla: solo mejoras en ground-truth accuracy).
- **Benchmark ampliado food-bench-v1** (31 clases × 8, Wikimedia Commons, licencias CC0/CC BY/CC BY-SA/PD): top1 16.1% con YOLO-only + threshold 0.20, 55.6% unknown. La detección YOLO falla 13/16 en fries/fried_chicken de Commons (fondos complejos) → el DINO fallback es esencial en fotos reales.
- **Dedup espacial (FASE 17)**: cap 1/clase reemplazado por IoU>0.3 + contención>70%: instancias separadas de la misma clase sobreviven (2 cookies ✓); cajas envolventes de DINO (plato) se colapsan. Multi-food: dup 0→4/16 (2 son instancias reales que el GT por clase cuenta como dup), recall 57.1% igual, precisión 54.5%→46.2% (métrica por clase subestima multi-instancia).
- **Threshold**: 0.20 mantenido (insensible 0.15-0.25 en food-us; en food-bench-v1 los crops YOLO dan scores binarios).
