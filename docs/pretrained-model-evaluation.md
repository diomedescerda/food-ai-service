# Evaluación de modelos preentrenados (FASE 10.5 — Research/Benchmark)

Fecha: 2026-08-28. Resultados reales en `benchmark_results.json`.
Benchmark: 108 imágenes (Wikimedia Commons, licencias CC, metadata FASE 10),
6 clases: french_fries, fried_chicken, hamburger, hot_dog, pizza, sandwich.

## Resultados medidos (CPU, sin GPU)

| Modelo | Top-1 | Top-3 | Top-5 | Latencia | Cobertura | Licencia |
|---|---|---|---|---|---|---|
| YOLO11n COCO (actual) | 0.370 | — | — | 112 ms | 0.556 (solo 3/6 clases) | AGPL-3.0 |
| CLIP ViT-B/32 (zero-shot, 34 clases) | **0.685** | 0.843 | 0.880 | 92 ms | 1.0 (catálogo ampliable) | MIT |
| SigLIP base (zero-shot) | 0.046 | 0.148 | 0.231 | 211 ms | 1.0 | Apache-2.0 |

## Multi-food (obligatorio)

Combo sintético pizza+fries: CLIP top5 = [eggs, pizza, pancakes, mashed
potatoes, sandwich] — **"french fries" NO aparece**. El embedding global de la
imagen completa predice una sola clase dominante y pierde el segundo alimento.
Conclusión: para imágenes con múltiples alimentos se necesitan **crops por
detección/segmentación** antes del embedding/zero-shot. El pipeline híbrido
(abajo) lo resuelve reutilizando YOLO11n-seg para crops y CLIP para identidad.

## Recomendación (híbrida)

```
Image
  ↓
YOLO11n-seg (detección + máscaras; reutilizado)
  ↓
crops por alimento
  ↓
CLIP ViT-B/32 zero-shot (catálogo ampliable, 34+ clases, sin reentrenar)
  ↓
Food Catalog + pgvector (futuro, si el catálogo crece a cientos)
  ↓
Nutrition DB (existente, fuente de verdad)
  ↓
Portion Engine (existente)
```

- CLIP reemplaza a `DetectorBasedClassifier` como `ZeroShotFoodClassifier`
  (misma interfaz IFoodClassifier — sin romper el pipeline).
- YOLO sigue siendo el detector (bbox/máscaras); CLIP da identidad con
  cobertura amplia y catálogo extensible (agregar clase = agregar texto).
- pgvector: NO necesario hoy (34 clases, catálogo corto); se añade si el
  catálogo llega a cientos de entradas y el matching por texto no basta.
- VLM (GPT-4o-mini/Claude Haiku/Gemini Flash): viable como FASE 9 posterior
  (platos compuestos, descomposición) a ~$0.0004/imagen (GPT-4o-mini 2026) —
  NO probado localmente (CPU inviable en tiempo razonable); marcado NOT TESTED.

## ¿Entrenar/dataset propio?

- TRAINING FROM SCRATCH: NO.
- FINE-TUNING: NO por ahora — CLIP zero-shot cubre más con cero esfuerzo de
  entrenamiento; se re-evalúa si la precisión de producción no alcanza.
- Dataset propio: NO como requisito para MVP de reconocimiento; el MVP actual
  (108 imágenes) queda como **benchmark de regresión** y futuro set de
  evaluación de calidad, no de entrenamiento.

## Licencias

CLIP (openai/clip-vit-base-patch32): MIT — uso comercial ✓.
SigLIP: Apache-2.0 ✓. YOLO11n: AGPL-3.0 (detector existente, revisar antes de
distribuir; el uso interno dev sigue igual).

## Privacidad

CLIP/SigLIP/YOLO = inferencia LOCAL (la imagen nunca sale del servicio).
VLM cloud enviaría la foto a un tercero (requiere consentimiento — FASE futura).