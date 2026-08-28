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

## FASE 5 — Nutrition Database ✅ (2026-08-27)
- [x] Fuente: USDA FoodData Central (dominio público, sin licencia restrictiva), versión 2026-08-27
- [x] Schema `foodai.` en PostgreSQL compartido: foods + food_nutrition (numeric(10,2), por 100 g) + food_aliases
- [x] Migración `AddFoodAiNutrition` + GRANT app_user
- [x] Seed idempotente `FoodAiNutritionSeeder` (9 alimentos; sandwich sin entrada — documentado, no inventado)
- [x] `INutritionProvider` → `DatabaseNutritionProvider` (alias o nombre canónico → nutrición por 100 g)
- [x] Endpoint verificación: `GET /api/v1/foodai/nutrition/{foodKey}` (404 controlado)
- [x] Separación estricta IA ↔ nutrición (la IA nunca escribe valores)
- [x] Tests: 200 unit backend (2 endpoint nuevos), 3 integration (PG real: lookup, null, seed reproducible)

## FASE 6 — Basic Portion Estimation ✅ (2026-08-27)
- [x] `IPortionEstimator` + `BasicPortionEstimator` (referencia FDC + heurística visual; separado del pipeline)
- [x] small/medium/large/unknown con min/estimated/max grams + confidence 0.55 (separada de detección) + method
- [x] Referencias documentadas (docs/portion-estimation.md): medidas domésticas USDA FDC
- [x] UNKNOWN sin inventar: sandwich (sin referencia) → null + confidence 0
- [x] Sin depth/volumen/densidad (FASE 7 pendiente)
- [x] Backend .NET: PortionDto + cliente parsea porción
- [x] Tests: 40 pytest (10 estimator + 3 real), 200 unit backend
- [x] E2E real: pizza 128 g (large), banana 118 g (medium), apple 218 g (large)

## FASE 7 — Advanced Portion Estimation ✅ (2026-08-28) — veredicto: sin escala física
- [x] Selección: Depth Anything V2 Small (Apache-2.0) — docs/depth-model-selection.md
- [x] `IDepthEstimator` + `DepthAnythingEstimator` (carga única, ~13 s; 0.76-0.86 s/img)
- [x] Depth map real + debug visual (jet) en 3 imágenes (6 archivos debug)
- [x] `PortionGeometryEstimator`: min/max/mean/median/p25/p75 + contraste máscara vs fondo
- [x] `AdvancedPortionEstimator`: sin escala física → method advanced_depth_relative, gramos NULOS (nunca masa inventada)
- [x] Health: depth_model {loaded, version}
- [x] Config: FOOD_AI_PORTION_METHOD basic|advanced + FOOD_AI_DEPTH_* — comparación sin tocar código
- [x] Señal real: contraste banana 0.770, apple 0.747, pizza 0.119
- [x] Conclusión: profundidad relativa NO mejora gramos sin escala → no recomendado para masa; feature geométrica para futuro
- [x] Tests: 49 pytest (9 nuevos depth/geometry/advanced), 200 unit backend (sin cambios de contrato — retrocompatible)

## FASE 8 — Nutrition Engine ✅ (2026-08-28)
- [x] Decisión: cálculo en .NET (la Nutrition DB vive en PostgreSQL — única fuente; sin duplicar en Python)
- [x] `INutritionCalculator` + `NutritionCalculator`: nutrient × grams / 100, decimal, redondeo display 2
- [x] Integración handler: por alimento → provider (100 g) → calculadora → nutrition + nutritionRange (min/max gramos)
- [x] summary + summaryRange (totales; alimentos sin nutrición excluidos, visibles individualmente)
- [x] food not found → nutritionStatus "unavailable"; sin gramos → "portion_unavailable" (no inventa, no rompe análisis)
- [x] source/sourceVersion en la respuesta
- [x] Sin doble contabilización (una entrada por detección del pipeline)
- [x] API /analyze retrocompatible + nutrition/summary (controller mapea presentación)
- [x] Tests: 211 unit backend (11 calculator nuevos) · E2E real: pizza 340.48 kcal, banana 105.02, apple 113.36

## FASE 9 — Food Analysis / User Feedback ✅ (2026-08-28)
- [x] Persistencia en schema foodai: food_analyses + food_analysis_items + food_analysis_feedback (migración AddFoodAiAnalysis + GRANT)
- [x] Snapshot nutricional + model versions por análisis (reconstruible aunque cambie el modelo/DB)
- [x] Máscaras a object storage (maskKey), nunca base64 en PG
- [x] Idempotencia por analysisId (AddAsync no duplica)
- [x] GET /analyses/{id} [Authorize] + ownership (404 análisis ajenos; 401 sin token)
- [x] POST feedback: FOOD_WRONG/PORTION_WRONG/DETECTION_WRONG/MISSING_FOOD/OTHER — conserva original + corregido
- [x] E2E real: analyze autenticado → GET → feedback (128→200) → GET (original intacto) → 401 sin token
- [x] Tests: 213 unit + 14 integration (4 persistencia nuevos)
- [x] Causa raíz login: admin solo con app "erp" (auth.user_applications) → login con application="erp" (sin tocar código)

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
## FASE 10B — Zero-shot classification (CLIP) ✅ (2026-08-28)
- [x] ZeroShotFoodClassifier (IFoodClassifier): crops de YOLO → CLIP ViT-B/32 → catálogo 39 entradas → top-k + threshold UNKNOWN
- [x] A/B configurable: FOOD_AI_CLASSIFIER_TYPE detector_based|zero_shot
- [x] Health: classifier_model clip-zero-shot-v1 · E2E: pizza 0.30, banana 0.34
- [x] Benchmark A/B (108 img): hybrid 0.491 global / 73.6% top-1 sobre crops vs YOLO 0.37; det 88ms + clip 86ms
- [x] Limitación: 36/108 sin crop (detección YOLO limita recall del pipeline)
- [x] Tests: 65 pytest (12 unit zeroshot + 4 real CLIP)

## FASE 11 — Food Region Detection ✅ (2026-08-28)
- [x] GroundingDinoDetector (open-vocabulary, prompt 'food on a plate') + HybridFoodDetector (YOLO → DINO fallback)
- [x] Config: FOOD_AI_DETECTOR_TYPE yolo|dino|hybrid
- [x] Region recall (108 img): YOLO 66.7% (72/108) vs DINO 100% (108/108)
- [x] E2E CLIP: YOLO+crops 0.491 vs DINO+crops 0.694 Top-1 (108/108 crops)
- [x] Latencia: YOLO 114ms · DINO 10.3s · híbrido ~3.4s media (fallback solo sin YOLO)
- [x] E2E real: pizza (yolo) 0.30; hamburger_000 (dino fallback) genera regiones → CLIP
- [x] Tests: 65 pytest sin regresión
