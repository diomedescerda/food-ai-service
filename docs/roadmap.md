# Roadmap — Food AI

Orden obligatorio: cada fase termina con tests + docs + commit. No avanzar con fase rota.

## FASE 0 — Infraestructura base ✅ (2026-08-27)
- [x] Backend: `IFoodAiClient` + `FoodAiController` (`GET /api/v1/foodai/health`) en coppAddresdBack
- [x] `food-ai-service/`: FastAPI + `/health` (puerto 8010) + tests
- [x] docker-compose raíz CoppAddresd (food-ai-service)
- [x] Backend → AI Service probe real verificado
- [x] Tests: 179 unit backend ✓, 2 pytest ✓

## FASE 1 — Image Ingestion ✅ (2026-08-27)
- [x] `POST /api/v1/foodai/analyze` (multipart) — MediatR command `AnalyzeFoodImageCommand`
- [x] Validación: extensión, MIME, tamaño máx (10 MB), vacío, firma mágica (`ImageFileValidator`)
- [x] `AnalysisId` UUID por análisis
- [x] `IImageStorage` → `LocalImageStorage` (delega en `IObjectStorageService` Local/S3 existente, clave `foodai/<id>.<ext>`)
- [x] Contrato backend → Python: `POST /analyze` multipart (`image` + `analysis_id`) → `{analysis_id, status: "received"}`
- [x] `FoodAiClient.SendImageAsync` (multipart, snake_case, errores → `FoodAiException` 502)
- [x] Correlación: `CorrelationIdDelegatingHandler` existente
- [x] E2E real verificado: validación + storage + respuesta received
- [x] Tests: 198 unit backend ✓ (19 nuevos), 9 pytest ✓
- [x] Sin tablas nuevas (persistencia síncrona; análisis persistidos en FASE 16 con correcciones)

## FASE 2 — Food Detection ✅ (2026-08-27)
- [x] Selección de modelo documentada (`docs/model-selection.md`): YOLO11n (Ultralytics, COCO, AGPL-3.0)
- [x] `IFoodDetector` + `YoloFoodDetector` (carga única en startup, filtro a 10 clases food COCO)
- [x] `POST /analyze` → `{status:"completed", model_version, inference_time_ms, foods[]}` con bbox en píxeles
- [x] Threshold configurable (0.35), imgsz 640, device, max_det — sin hardcode
- [x] Debug visual: `utils/debug.py` overlay bbox+labels+confidence
- [x] Model versioning: `food-detector-v1` en cada respuesta
- [x] Backend: DTOs `DetectedFood/BoundingBox`, cliente parsea `foods`, tests actualizados
- [x] Tests: 17 pytest (incl. 4 YOLO real: carga única, sin falsos positivos), 198 unit backend
- [x] Dockerfile: ultralytics + modelo en build (cache), volumen futuro
- [x] E2E real: backend → food-ai → YOLO → detecciones

## FASE 3 — Segmentation ✅ (2026-08-27)
- [x] `IFoodSegmenter` + `YoloFoodSegmenter` (yolo11n-seg.pt) — interfaces separadas del detector
- [x] `/analyze` → `segmentation` por alimento (PNG b64 recortado al bbox + area_pixels) + `seg_model_version`
- [x] Emparejamiento máscara↔detección por IoU ≥ 0.5
- [x] Debug overlay: máscara semi-transparente + bbox + label + confidence
- [x] Health: `segmentation_model` con loaded/version
- [x] Backend .NET: `SegmentationDto` + cliente parsea máscara
- [x] Tests: 22 pytest (incl. 3 segmentación real pizza/banana/apple: dims=bbox, área>0, área<bbox), 198 unit backend
- [x] E2E real por backend: pizza 0.9253/44970px, banana 0.8932/24629px

## FASE 4 — Classification ✅ (2026-08-27)
- [x] Decisión: clasificación del detector suficiente (YOLO = detector+clasificador integrado); SIN segundo modelo
- [x] `IFoodClassifier` + `DetectorBasedClassifier` (delega en detector, costo 0 ms) — pipeline contra interfaces
- [x] `/analyze` → `classifier_version` (retrocompatible)
- [x] Health: `classifier_model` {loaded, version}
- [x] Backend .NET: `ClassifierVersion` en DTO/cliente
- [x] Tests: 27 pytest (5 nuevos: fake + clasificación real pizza/banana/apple), 198 unit backend
- [x] E2E real: pizza 0.9253, banana 0.8932, apple 0.8613 — coherentes
- [x] docs/classification.md: decisión, límites, plan comida colombiana

## Pendiente (orden)

1. **FASE 1 — Image ingestion**: `POST /api/food/analyze` — validación MIME/tamaño, id, almacenamiento temporal, reenvío al AI Service, respuesta asíncrona.
2. **FASE 2 — Food detection**: modelo preentrenado (YOLO) tras `IFoodDetector`; bounding boxes + confidence.
3. **FASE 3 — Segmentación**: máscaras por alimento + imagen debug (original + boxes + masks + labels).
4. **FASE 4 — Clasificación**: capa dedicada (preparar `GenericFoodClassifier` → `ColombianFoodClassifier`).
5. **FASE 5 — Nutrition DB**: tablas (schema `foodai.`), `NutritionService` determinista.
6. **FASE 6 — Porción básica**: small/medium/large → rangos de gramos; `PortionEstimator` abstracto.
7. **FASE 7 — Porción avanzada**: depth estimation/área/densidad, solo con métrica previa (MAE).
8. **FASE 8 — Nutrition engine**: food + grams + BD → macros + totales.
9. **FASE 9 — Vision LLM auxiliar**: candidatos desde LLM → catálogo → BD (nunca fuente final).
10. **FASE 10 — Dataset colombiano**: 30–50 clases (arroz, arepa, frijoles, bandeja paisa…).
11. **FASE 11 — Anotación/dataset pipeline**: `prepare/validate/split_dataset.py`, licencias documentadas.
12. **FASE 12 — Fine-tuning**: métricas (mAP, IoU, F1), versionado `food-model-vN`.
13. **FASE 13 — Confidence system**: detección/clasificación/porción/nutrición separadas.
14. **FASE 14 — Correcciones de usuario**: original vs corrección.
15. **FASE 15 — Feedback dataset**: correcciones → validación → dataset.
16. **FASE 16 — API final**: analyze/analysis/{id}/correction/foods/nutrition + OpenAPI.
17. **FASE 17 — Frontend**: UI en `coppaddresd-front/` (subir foto, resultados, corregir, "Valores estimados").
18. **FASE 18 — Testing**: unit/integration/E2E + set de imágenes de regresión.
19. **FASE 19 — Evaluación**: benchmark (detección, clasificación, IoU, porción MAE, kcal MAE).
20. **FASE 20 — Producción**: Docker images, CI/CD, S3, RDS, escalamiento, observabilidad.

## Reglas transversales

- Modelos/datasets/checkpoints NUNCA en git.
- Versión de modelo en cada análisis (`modelVersion`).
- LLM nunca calcula macros directo.
- Fase nueva solo tras autorización del usuario.