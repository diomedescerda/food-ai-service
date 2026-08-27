# AI Pipeline — Food AI

Documento de diseño del pipeline de inferencia. **FASE 0: sin pipeline implementado.** Este doc registra el diseño acordado para las fases 2–9.

## Pipeline objetivo

```
imagen
  ↓
detección de alimentos          (FASE 2, modelo preentrenado YOLO tras IFoodDetector)
  ↓
segmentación                    (FASE 3, máscaras por alimento + imagen debug)
  ↓
clasificación                   (FASE 4, capa dedicada: Generic → ColombianFoodClassifier)
  ↓
estimación de porción           (FASE 6-7, híbrido: small/medium/large → grams; luego depth/área/densidad)
  ↓
identificación nutricional      (FASE 5, BD determinista, NUNCA IA)
  ↓
cálculo nutricional             (FASE 8, motor: food + grams → macros)
  ↓
resultado + confianza           (FASE 13)
```

## Principios

- Componentes reemplazables: `IFoodDetector`, `IClassifier`, `ISegmenter`, `IPortionEstimator`, `INutritionProvider`.
- El cálculo nutricional es determinista: `100 g arroz = 130 kcal` escalado por gramos. La IA jamás inventa macros.
- El LLM (FASE 9) solo propone candidatos que pasan por el catálogo y la BD.
- Estimaciones de porción se comunican con confianza, sin falsa precisión.

## Métricas de evaluación (FASE 19)

Detección accuracy · clasificación accuracy · segmentación IoU · porción MAE · kcal/carbos MAE.