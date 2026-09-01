# EvaluaciÃ³n de modelos preentrenados (FASE 10.5 â€” Research/Benchmark)

Fecha: 2026-08-28. Resultados reales en `benchmarks/history/benchmark_results.json`.
Benchmark: 108 imÃ¡genes (Wikimedia Commons, licencias CC, metadata FASE 10),
6 clases: french_fries, fried_chicken, hamburger, hot_dog, pizza, sandwich.

## Resultados medidos (CPU, sin GPU)

| Modelo | Top-1 | Top-3 | Top-5 | Latencia | Cobertura | Licencia |
|---|---|---|---|---|---|---|
| YOLO11n COCO (actual) | 0.370 | â€” | â€” | 112 ms | 0.556 (solo 3/6 clases) | AGPL-3.0 |
| CLIP ViT-B/32 (zero-shot, 34 clases) | **0.685** | 0.843 | 0.880 | 92 ms | 1.0 (catÃ¡logo ampliable) | MIT |
| SigLIP base (zero-shot) | 0.046 | 0.148 | 0.231 | 211 ms | 1.0 | Apache-2.0 |

## Multi-food (obligatorio)

Combo sintÃ©tico pizza+fries: CLIP top5 = [eggs, pizza, pancakes, mashed
potatoes, sandwich] â€” **"french fries" NO aparece**. El embedding global de la
imagen completa predice una sola clase dominante y pierde el segundo alimento.
ConclusiÃ³n: para imÃ¡genes con mÃºltiples alimentos se necesitan **crops por
detecciÃ³n/segmentaciÃ³n** antes del embedding/zero-shot. El pipeline hÃ­brido
(abajo) lo resuelve reutilizando YOLO11n-seg para crops y CLIP para identidad.

## RecomendaciÃ³n (hÃ­brida)

```
Image
  â†“
YOLO11n-seg (detecciÃ³n + mÃ¡scaras; reutilizado)
  â†“
crops por alimento
  â†“
CLIP ViT-B/32 zero-shot (catÃ¡logo ampliable, 34+ clases, sin reentrenar)
  â†“
Food Catalog + pgvector (futuro, si el catÃ¡logo crece a cientos)
  â†“
Nutrition DB (existente, fuente de verdad)
  â†“
Portion Engine (existente)
```

- CLIP reemplaza a `DetectorBasedClassifier` como `ZeroShotFoodClassifier`
  (misma interfaz IFoodClassifier â€” sin romper el pipeline).
- YOLO sigue siendo el detector (bbox/mÃ¡scaras); CLIP da identidad con
  cobertura amplia y catÃ¡logo extensible (agregar clase = agregar texto).
- pgvector: NO necesario hoy (34 clases, catÃ¡logo corto); se aÃ±ade si el
  catÃ¡logo llega a cientos de entradas y el matching por texto no basta.
- VLM (GPT-4o-mini/Claude Haiku/Gemini Flash): viable como FASE 9 posterior
  (platos compuestos, descomposiciÃ³n) a ~$0.0004/imagen (GPT-4o-mini 2026) â€”
  NO probado localmente (CPU inviable en tiempo razonable); marcado NOT TESTED.

## Â¿Entrenar/dataset propio?

- TRAINING FROM SCRATCH: NO.
- FINE-TUNING: NO por ahora â€” CLIP zero-shot cubre mÃ¡s con cero esfuerzo de
  entrenamiento; se re-evalÃºa si la precisiÃ³n de producciÃ³n no alcanza.
- Dataset propio: NO como requisito para MVP de reconocimiento; el MVP actual
  (108 imÃ¡genes) queda como **benchmark de regresiÃ³n** y futuro set de
  evaluaciÃ³n de calidad, no de entrenamiento.

## Licencias

CLIP (openai/clip-vit-base-patch32): MIT â€” uso comercial âœ“.
SigLIP: Apache-2.0 âœ“. YOLO11n: AGPL-3.0 (detector existente, revisar antes de
distribuir; el uso interno dev sigue igual).

## Privacidad

CLIP/SigLIP/YOLO = inferencia LOCAL (la imagen nunca sale del servicio).
VLM cloud enviarÃ­a la foto a un tercero (requiere consentimiento â€” FASE futura).
