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

## FASE 12 — Nutrition Catalog Expansion ✅ (2026-08-28)
- [x] Catálogo curado food_usda_curated.json: 38 alimentos, 20 con nutrición verificada USDA (FDC IDs), 15 REVIEW_REQUIRED, 3 sin match
- [x] Seeder v2 idempotente lee el JSON (mapping_status/confidence/source_id en foods/food_nutrition)
- [x] Migración AddFoodAiNutritionMapping (source_id, mapping_status, mapping_confidence)
- [x] scripts/import_usda_foods.py: modo curated + modo API (FOODAI_USDA_API_KEY) para escalar a 100+
- [x] E2E: pizza 340.48 kcal (available), hamburger 297 kcal fdc_id=170693, steak FOOD_NOT_FOUND (review)
- [x] Tests: 213 unit + 14 integration sin regresión

## FASE 13 — End-to-End Nutrition Coverage ✅ (2026-08-28)
- [x] Métrica de producto medida: E2E Success Rate = 28.7% (31/108)
- [x] Desglose: detection 100% · classification 64.8% · mapping 38.9% · porción 28.7%
- [x] Error taxonomy: CLASSIFICATION_WRONG 38 · NUTRITION_MAPPING_MISSING 28 · PORTION_UNAVAILABLE 11
- [x] Cuello de botella: clasificación (CLIP) en hot_dog/fries/fried_chicken/sandwich; intento de prompt mejora falló (63%) → revertido (decisión por datos)
- [x] Porción: +hamburger 78g (FDC 170693 verificado); fries/fried_chicken sin referencia verificada → GAP
- [x] Multi-food estructural ✓ (hamburger_000 → 4 regiones); combo sintético falló detección (documentado)
- [x] Sin doble conteo: hybrid YOLO→DINO (test unit) · Tests: 67 pytest (2 nuevos)

## FASE 14 — CLIP Catalog Tuning ✅ (2026-08-28)
- [x] 13 experimentos controlados (prompts, ensemble, específicos, jerárquico, padding, masked, threshold) — resultados en benchmark_clasification_tuning/
- [x] Best: ensemble mean + padding 0.10 + thr 0.20 → Top-1 0.685 (+0.037) · E2E 30.6% (+1.9)
- [x] Descartados con evidencia: prompts específicos, jerárquico, masked crop
- [x] Confusion matrix + per-class + top-k analysis documentados
- [x] Tests: 67 pytest sin regresión

## FASE 15 — Product Gaps ✅ (2026-08-28)
- [x] Porciones fast food verificadas USDA (20 referencias nuevas: fries 117g FDC 170698, fried_chicken 203g 170756, taco 69g, burrito 185g, quesadilla 157g, nachos 80g, nuggets 87g, eggs 61g, bacon 12g, pancakes 77g, toast 22g, bagel 105g, waffles 75g, oatmeal 234g, rice 158g, pasta 124g, salad 35g, steak 85g, salmon 170g, ice_cream 66g)
- [x] E2E 108: 30.6% → 40.7% (+10.1 — fries/fried_chicken completan el pipeline)
- [x] Benchmark multi-food real: 16 imágenes de platos compuestos (Wikimedia, GT manual por nombre), dataset multi-food/
- [x] DEDUP: NMS IoU 0.5 en regiones DINO + máx 1 predicción por clase → duplicados 24→0, precisión 27.7%→54.5%
- [x] food_coverage_report.json: 38 clases, 19 con e2e_supported (50%)
- [x] USDA API KEY NOT AVAILABLE → 15 mappings pendientes sin inventar
- [x] Tests: 67 pytest sin regresión

## FASE 16 — Production Hardening ✅ (2026-08-28)
- [x] Fix cadena de aliases: provider .NET normaliza _→espacio (hot_dog→'hot dog'), estimator normaliza, porción usa identidad CLIP → hot_dog 0→13/20
- [x] E2E: 40.7% → 52.8% (57/108) — 3 bugs de alias encontrados y corregidos
- [x] Scoring por clase (max candidato): hot_dog recuperado (13/20) sin perder global (68.5%)
- [x] Threshold sweep E2E 0.15-0.25: insensible (52.8%) → se mantiene 0.20
- [x] Observabilidad: logs estructurados por análisis (tiempos, foods, nutricion_ok, fallos) sin datos de usuario
- [x] E2E real: hot_dog_001 → hot_dog 0.2387, porción 68g, 197.20 kcal available (USDA)
- [x] Tests: 69 pytest (2 nuevos normalización/score-clase), 213 .NET, 14 integration

## FASE 17 — Production Telemetry & Expanded Catalog ✅ (2026-08-28)
- [x] USDA API key NO disponible → 15 mappings siguen REVIEW_REQUIRED (no inventar)
- [x] Audit catálogo completo: 38 alimentos, 20 nutrition (52.6%), 30 portion (78.9%), 20 e2e_supported (52.6%) — scripts/audit_catalog.py
- [x] Benchmark ampliado food-bench-v1: 248 imágenes, 31 clases, Wikimedia Commons con licencias CC0/CC BY/CC BY-SA/PD + metadata.json completo
- [x] Classification v2: top1 16.1% (YOLO-only, fotos reales); la detección es el cuello en fotos variadas (DINO esencial)
- [x] Crop experiments fries/fried_chicken: padding y masked NO ayudan → confusión estructural CLIP
- [x] Candidate groups específicos: NET ZERO → rechazados
- [x] Telemetry: tiempos por etapa + used_dino_fallback en logs; DINO fallback rate real = 33.3%
- [x] Latencia: YOLO p50 78ms/p95 115ms; DINO p50 11.8s/p95 13.8s
- [x] Memoria real: WorkingSet 0.61GB / Private 2.12GB (modelos cargados)
- [x] Multi-instancia: dedup espacial (IoU 0.3 + contención 70%) — 2 cookies separadas ✓
- [x] Regresión protegida: E2E 52.8% sin cambio, hot_dog 13/20, fries 7/20
- [x] Tests: 72 pytest, 213 .NET, 14 integration

## FASE 18 — Nutrition Data Completion ✅ (2026-08-31)
- [x] USDA API key NO disponible → sin mappings nuevos; 15 bloqueados documentados (audit_usda_mappings.py)
- [x] Auditoría 20 mappings: sin duplicados, valores numéricos, trazabilidad vía seeder (source/sourceVersion/sourceId por item en BD); data_type no se usa (deuda)
- [x] GT multi-food por instancia (metadata.json: ground_truth_foods con instance_id) — evidencia espacial F17
- [x] Benchmark por instancia: recall 56.5% (13/23), precision 48.1%, missed 10, FP 14, dups 2
- [x] Doble conteo VERIFICADO en E2E real: mf_003 hamburger×2 → 2×184.14 = 368.28 kcal summary (antes colapsaba a 1)
- [x] Rangos validados: min<=estimated<=max y nutritionRange.min<=cal<=max en pizza/hot_dog/hamburger×2
- [x] Telemetría: nutrition_items_* en log del backend (total/available/unavailable/mapping_missing/ambiguous/calc_failed)
- [x] E2E real: pizza 228.76, hot_dog 197.20, banana 105.02, mf_003 368.28 (2 instancias); pancakes→steak (clasificación)
- [x] Benchmarks sin regresión: food-us 52.8%, v1 16.1%, tests 72/72, .NET 213/213, integration 14/14

## FASE 19 — USDA Nutrition Completion + Real Benchmark Validation ✅ (2026-08-31)
- [x] USDA key NO disponible → mappings bloqueados 🔒; importador --sync validado
- [x] FOOD-BENCH-V1 HYBRID (248, 31 clases): top1 31.9% (vs 16.1% YOLO-only); DETECTION FAILURE 0%, CLASSIFICATION FAILURE 68.1%
- [x] Separación con datos: el 16.1% de F17/18 era DETECCIÓN (YOLO falla en fotos reales), no clasificación
- [x] Fallback rate real benchmark: 55.6% (138/248) — YOLO 44.4%
- [x] Latencia: YOLO puro ~94-131ms, DINO ~10-11.6s, CLIP ~70-99ms por crop
- [x] Tabla por clase (21 clases): pizza 87.5%, fried_chicken 75%, rice 75%, cake 75%, nachos 62.5%...
- [x] Casos A-G: A (1 burger → 3 regiones: triple conteo mf_000 599.94 kcal — limitación 2D documentada); B (mf_003 2×184.14 ✓); C/E sintéticos fallan por clasificación (honesto)
- [x] E2E reales 7: pizza/hot_dog/banana/apple ✓ rangos OK; hamburger_000 → sandwich (clasificación); fries_000 → fried_chicken+apple
- [x] Tests: 72/72, 213/213, 14/14; food-us 52.8%; v1 YOLO-only 16.1%; v1 hybrid 31.9%; multifood 56.5%/48.1% F1 52.0%

## FASE 20 — USDA Unblock + Pretrained Food Model Evaluation ✅ (2026-08-31)
- [x] USDA KEY: DISPONIBLE en .env — el script no cargaba .env (os.environ sin dotenv) → fix load_dotenv en import_usda_foods.py
- [x] --sync OK: candidatos FDC reales; selección manual de 15 mappings genéricos (FNDDS/SR Legacy, sin marca) con nutrientes verificados vía API
- [x] Nutrition Coverage: 20/38 (52.6%) → 35/38 (92.1%): 21 DIRECT_MATCH + 14 GOOD_EQUIVALENCE; REVIEW_REQUIRED 15→0; NO_RELIABLE 3
- [x] Idempotencia ✓ (2º import idéntico); audit sin duplicados; seeder '15 nuevos de 38' ✓
- [x] Porciones nuevas (API foodPortions): lasagna 206g, mac_and_cheese 189g, grilled_chicken 120g, cookie 16g, brownie 50g → 35/38 con porción
- [x] E2E real nuevos: nachos 170.24 kcal, chicken_nuggets 267.09, salmon 261.80 ✓
- [x] A/B/C clasificadores (crops idénticos 355): CLIP food-us 58.3%/v1 43.3% top1, unknown 0%; DINOv3 55.6%/32.8% (19-42% unknown); BEiT 32.4%/25.9% (45-50% unknown) → CLIP GANA (también latencia 120ms vs 700-900ms)
- [x] Licencias: DINOv3 apache-2.0, BEiT MIT (ambas comerciales OK); modelos evaluados con medición propia (no model card: DINOv3 publica 90.6% en TSOTSA-Img, en nuestros benchmarks 55.6%)
- [x] Decisión: mantener CLIP (ningún modelo supera); documentado qué falta para el siguiente salto
- [x] Tests: 72/72, 213/213, 14/14

## FASE 21 — Production Food Identity + Dataset Integration ✅ (2026-08-31)
- [x] Branch feature/production-food-identity (checkpoint bee2b6e/58b9b41) — datasets KEEP: food-us, v1, multi-food
- [x] Food-101 estudiado (101 clases × 250 test, NON-COMMERCIAL → evaluación interna) — subset 3500 imágenes / 14 clases del catálogo
- [x] Nutrition5k estudiado (CC BY 4.0 ✓ comercial, RGB-D + masa por ingrediente): documentado como futura mejora de porción (requiere gsutil/GCS; no es mejora rápida demostrable esta fase)
- [x] Identidad CLIP Food-101: top1 79.7% → 89.7% (con catálogo + ensemble) — fallos sistemáticos detectados: taco 11.6% (→quesadilla), waffles 3.6% (→pancakes)
- [x] A/B dirigido (Food-101): candidates descriptivos taco/quesadilla/waffles/pancakes: waffles +88.8 pts, taco +53.6 (sin 'taco al pastor' que rompía hot_dog E2E)
- [x] v1 (fotos reales): top1 43.3% → 47.0% (+3.7); food-us E2E 52.8% protegido (hot_dog 13/20)
- [x] Dedup centro-contención (FASE 21): mf_000 mismo objeto fragmentado colapsa (dups 2→1, precision 48.1→50.0) sin romper instancias reales (mf_003 2 burgers, mf_006 2 eggs ✓)
- [x] E2E reales 14: pizza/hot_dog/salmon/nachos/lasagna/mac_and_cheese ✓ con nutrición; multi-food 2 instancias ✓
- [x] Benchmarks no regresionados: pytest 72/72, .NET 213/213, integration 14/14

## FASE 22 — Final Food Identity Optimization + Production Readiness ✅ (2026-08-31)
- [x] Branch feature/final-food-identity (checkpoint 4198d1b); datasets KEEP (food-us, v1, multi-food, food101-subset)
- [x] P1 fries/fried_chicken: crop inset A/B (0.0-0.15) NO mejora (fries 3/20 vs 7/20) — confusión rice es representacional; candidates 'thin fried potato strips'/'long golden potato sticks' SÍ (food-us E2E fries 7→13)
- [x] P2-P4 candidates descriptivos: hot_dog 'in a long bun', hamburger 'round sesame bun', sandwich 'between two slices', nachos 'melted cheese on tortilla chips', salmon 'pink flesh', steak 'grilled beef steak on a plate', grilled_chicken 'breast pieces'
- [x] E2E food-us: 52.8% → 59.3% (+6.5): fries 13/20 (+6), hot_dog 14/20 (+1), hamburger 16/20 (+1), pizza 17/20, fried_chicken 4/8
- [x] v1: 47.0% mantenido (clases objetivo mejoran per-clase); Food-101 89.4% (sin regresión)
- [x] Multi-food: precision 52.0% (+2), dups 1, doble conteo 2/2; mf_000 3→2 instancias (dedup centro-contención E2E: 599.94→415.80)
- [x] Telemetría completa en log: foods_detected/classified/unknown/instance_count + fallback + tiempos
- [x] Bug encontrado: backend se cuelga tras ~7-10 análisis consecutivos (sin log de error) — documentado para producción
- [x] E2E reales 21: pizza/banana/apple/hot_dog/fries/salmon/nachos/lasagna/mac/mf_003/mf_006/mf_010/mf_011 ✓ con nutrición y rangos; fallos de clasificación honestos (steak/taco/quesadilla img_0001, fried_chicken→rice)
- [x] Recomendación sandwich/soup/cereal: mantener con nutritionStatus=unavailable (respuesta honesta, sin inventar)
- [x] Tests: pytest 72/72, .NET 213/213, integration 14/14 — configuración de despliegue documentada

## FASE 23 — Production Stability, Load Testing & Deployment Gate ✅ (2026-08-31)
- [x] CAUSA RAÍZ del 'hang' de F22: NO era hang — el backend era TERMINADO externamente (otra sesión de agente editaba el backend en paralelo); sin crash en stderr ni Event Log. 100/100 secuenciales OK en entorno aislado
- [x] CONCURRENCIA: torch CPU crashea (nativo, sin traceback) con DINO bajo >=4 requests simultáneos → fix: asyncio.Semaphore(1) serializa la inferencia (app.state.inference_semaphore); conc 2: 50/50 (p50 10.9s), conc 4: 13/50 → 50/50 (p50 24.7s), conc 8: 37-50 OK con rechazos controlados del listener (p50 49s)
- [x] Load: 100 secuenciales 100/100, p50 451ms, p95 12.9s, p99 16.3s; backend WS estable (136MB); food-ai WS oscila (611→1294MB, no lineal); PG 1 conexión estable (sin fuga); HTTP sin fuga
- [x] DINO fallback: 33-56% según dominio; p50 11.8s; fast path p50 0.3-0.6s
- [x] Graceful failure: imagen inválida → 400; health/readiness: /health (modelos+servicio) + /api/v1/foodai/health (probe backend+foodAI) ✓
- [x] Regression gate: 72/72 pytest, 213/213 .NET, 14/14 integration; E2E clave sin regresión (pizza/hot_dog/salmon/nachos/fries/mf_003/mf_006/mf_011 ✓)
- [x] DECISIÓN: PRODUCTION READY con límites documentados (concurrencia <=4 por la serialización CPU; DINO 10-17s en 33-56% de requests)

## FASE 24 - Cleanup + Documentacion OK (2026-08-31)
- [x] Reorganizacion: JSON de resultados -> benchmarks/{classification,detection,nutrition,multifood,history}; regions.json -> benchmarks/detection/; basura eliminada (debug images, ab_progress)
- [x] README completo (34 secciones, 7 diagramas Mermaid, glosario) - documentacion oficial del servicio
- [x] OUTs de scripts actualizados a benchmarks/; refs de regions.json corregidas en 7 scripts
- [x] Sin regresion: pytest 72/72, E2E food-us 59.3%

## FASE 25 — Massive Food Catalog Expansion ✅ (2026-08-31)
- [x] Food Master Catalog (data/catalogs/food_master.json): 231 clases con estados (65 PRODUCTION_READY, 146 VISUAL_ONLY, 17 REVIEW_REQUIRED, 3 NO_RELIABLE)
- [x] 53 mappings USDA nuevos verificados (43 DIRECT + 10 GOOD) via API + 17 falsos revertidos (Cape Cod, Avocado dressing...) + porciones foodPortions
- [x] Nutrition5k documentado como estratégico (CC BY 4.0, masa real por ingrediente; descarga vía gsutil pendiente)
- [x] A/B escalamiento: CLIP scoring único con 465 candidates DEGRADA a 0% (baseline 71.3%) — los candidates descriptivos compiten (waffles 0.27 gana en pizza); el top_k no salva. Conclusión: se requiere CANDIDATE RETRIEVAL (coarse-to-fine) antes de activar el catálogo ampliado — pipeline de producción (38) intacto
- [x] 38 clases = regression set intacto (71.3% food-us con score por clase)

## FASE 26 — Coarse-to-Fine Candidate Retrieval ✅ (2026-08-31)
- [x] Retrieval en memoria implementado (CLIP image ↔ text embeddings de 231 clases, top-N → detailed ensemble por clase)
- [x] RECALL@N (métrica crítica): food-us @10 71.3% @20 80.6% @50 87.0% · v1 @10 55.1% @20 62.8% @50 76.1% · Food-101 @10 93.4% @20 97.0% @50 99.0% — el retrieval recupera bien
- [x] TOP-1 final (detailed sobre top-N): food-us 34.3% vs legacy 71.3% · v1 26.3% vs 47.0% · Food-101 49.6% vs 89.4% — el detailed NO supera legacy
- [x] DECISIÓN: NO en producción (top1 retrieval << legacy); legacy 38 = default. El retrieval resuelve la GENERACIÓN de candidatos (recall@N) pero el detailed necesita prototypes visuales (las 193 sin imágenes) — siguiente fase
- [x] Latencia retrieval: ~215-280ms vs ~70ms legacy (3× — matmuls 465 candidates ×2 pasos)
- [x] Regression set intacto (legacy default); tests 72/72

## FASE 27 — Visual Prototype Retrieval ✅ (2026-08-31)
- [x] Prototype store: 17 clases, 126 prototipos (food-bench-v1 Commons CC0/CC BY/CC BY-SA/PD + multi-food + assets — permisivas; naan sin imágenes → fallback texto)
- [x] Hybrid scoring (text retrieval top-10 + visual similarity, barrido w 0.25/0.5/0.75 × agg max/mean/topk5/medoid)
- [x] RESULTADO: food-us 60.2% (w0.25 topk5_mean) vs legacy 71.3% vs F26 34.3% · v1 37.2% (w0.5 max) vs 47.0% vs 26.3% · Food-101 74.9% (w0.25 medoid) vs 89.4% vs 49.6%
- [x] Los prototipos RECUPERAN ~26 pts del F26 pero NO superan el legacy → NO aprobar 231 en producción (Gates 1-2 no se cumplen)
- [x] Prototype store: 681 KB (126 × 512 floats); latencia hybrid ~215ms (misma que F26 — el visual es matmul trivial)
- [x] Conclusión: los prototipos visuales resuelven PARCIALMENTE F26; falta cobertura (17/231 clases) y calidad (8 prototipos por clase de Commons con ruido)

## FASE 28 — Prototype Coverage + Quality ✅ (2026-08-31)
- [x] Calidad: 24 WRONG/7 DUP de 126 (salmon 5/8 ruido); A/B raw vs clean IDENTICOS — la limpieza no cambia (agregación diluye)
- [x] Cobertura: 17 → 33 clases (254 prototipos, food-bench completo + assets); 198 clases INSUFFICIENT_DATA
- [x] RESULTADO: v1 55.1% SUPERA legacy 47.0% (+17.9 vs F27); Food-101 85.0% (94.8% del legacy); food-us 59.3% (gap −12: fries/fried→rice estructural)
- [x] DECISIÓN: REJECT 231 (food-us no alcanza legacy); la cobertura es la dirección correcta (evidencia v1); faltan 198 clases + resolver fries→rice

## FASE 29 — Prototype Scale + Rice Resolution ✅ (2026-08-31)
- [x] Rice analysis: hybrid enfocado w=0.5 -> 21/28 (75%) vs legacy 60.7% — la confusión rice se reduce con scoring dirigido (4->rice, 3->fried restantes)
- [x] Cobertura 33->68 (36 clases Commons nuevas): food-us 59.3 -> 38.9 DEGRADA; v1 55.1 -> 53.4 — la cobertura amplia NO escala (prototipos ruidosos compiten)
- [x] DECISIÓN: REJECT 231 — techo arquitectónico ~60% del prototype approach; detener expansión masiva; F30 = fine-tuning/representación o legacy

## FASE 30 — CLIP Embedding Adaptation + Confusion-Group Classifier ✅ (2026-08-31)
- [x] Frozen CLIP + Linear: grupo rice 85.7% (fries 20/20, elimina fries→rice) vs legacy 60.7% y F29 75% — la frontera aprendida SÍ separa el grupo
- [x] Global 67 clases (8 imgs/clase): Linear 17.6% / MLP 15.7% — no escala con datos mínimos (Caso E parcial)
- [x] JERÁRQUICO (legacy + grupo rice): 77.8% food-us > 71.3% legacy (+6.5) — Caso D: la vía es legacy + specialized groups, no el modelo global
- [x] DECISIÓN: APPROVE experimental para la arquitectura jerárquica (grupos de confusión); default = legacy; F31 implementa ConfusionGroup generalizado + más datos

## FASE 31 — Generalized Confusion-Group Hierarchical Classifier ✅ (2026-08-31)
- [x] ConfusionGroup abstraction: 5 grupos entrenados (Linear CLIP frozen) — SOLO rice habilita (75% vs legacy 60.7, +14.3); pizza/naan (40%), hamburger/sandwich (47.5%), taco (sin GT), fried/nuggets (50%) DEGRADAN -> enabled=false (criterio: accuracy > legacy)
- [x] JERÁRQUICO food-us (legacy + rice, crops): 72.2% > 71.3% legacy ✓ (Gate 1); 77.8% de F30 no se reproduce exactamente (variación por semilla con 24 imágenes de train — documentado)
- [x] DECISIÓN: REJECT default — la arquitectura valida el concepto; los otros grupos fallaron por DATOS (8 imgs/clase), no por el enfoque
- [x] F32: más datos por grupo (Nutrition5k/Commons) + gating fino + re-evaluar grupos con >=20 imgs

## FASE 32 — Data Sufficiency + Deterministic Splits + Gating ✅ (2026-08-31)
- [x] Splits deterministas (hash md5 % 20) + multi-seed (42/123/2026) + gating 0.6 — infraestructura lista
- [x] HALLAZGO: más datos CRUDOS de Commons degradan (rice 75% -> 46.4%) — el punto de saturación de rice es ~8-12 imágenes LIMPIAS; pizza/naan SÍ mejora con datos (40 -> 86.7)
- [x] Diagnóstico de representación: hamburger/sandwich MUY separable (inter 0.403) — el sandwich 0/20 es calidad del train; fried/nuggets NO separable (inter 0.637 ≈ intra) — representación
- [x] Global con datos crudos = legacy (71.3); F31 (limpio) sigue siendo el mejor (72.2)
- [x] DECISIÓN: REJECT — calidad > cantidad; F33 = pipeline de validación obligatorio + LoRA para representación-limitados

## FASE 33 — Validated Data + Selective Adaptation ✅ (2026-08-31)
- [x] Validación obligatoria: 344 candidatas -> 129 VALID (fries 0, naan 1, sandwich 3 — las clases de confusión eliminadas por el top-1 del legacy y el dedup)
- [x] GRUPOS con VALID: pizza/naan 96.7% ± 4.7 (vs 85 legacy) y fried/nuggets 100% ± 0 (vs 50 — la validación lo arregló: 33±47 crudo); LoRA 50% < frozen -> descartado
- [x] HALLAZGO: el flujo por predicción-legacy (prompt §8) solo CONFIRMA — el especializado no rescata las imágenes que legacy ya falló -> global 71.3 = legacy
- [x] rice degradó con la validación estricta (29.8) — las fries difíciles eliminadas
- [x] DECISIÓN: REJECT — el gating por legacy-confidence (activar especializado cuando legacy confía poco) es el siguiente paso (F34)

## FASE 34 — Confidence Gating + Rescue ✅ (2026-08-31)
- [x] Gating por confianza (T sweep 0.20-0.70 × S/M) + routing por top-3 del legacy
- [x] RESULTADO: RESCUE=0 en todos los thresholds — los errores de legacy y especialista están CORRELACIONADOS (mismo embedding CLIP); el top-3 activa grupos irrelevantes (harmful hasta 7)
- [x] DECISIÓN: REJECT + CIERRE del paradigma frozen-CLIP + confusion groups (criterio del prompt: falla incluso con especialistas 96.7%/100%)
- [x] F35: backbone visual más fuerte (DINOv2/SigLIP/CLIP-L) como representación del clasificador aprendido

## FASE 35 — Strong Visual Backbones ✅ (2026-08-31)
- [x] Benchmark: CLIP-B/32 vs DINOv2-L vs SigLIP (CLIP-L sin evaluar — disco)
- [x] DIAGNÓSTICO: DINOv2-L inter 0.038-0.218 vs 0.60-0.67 (separación 3-15x mejor) — la hipótesis de representación CONFIRMADA
- [x] Grupos: pizza/naan dino 100% (vs 88.3), fried/nuggets siglip 100%, hamburger/sandwich dino 49.2 (con 3 sandwich VALID); rice bloqueado (fries 0 VALID)
- [x] Global dino Linear: 56.4% (no supera legacy — datos, no representación)
- [x] DECISIÓN: APPROVE REPRESENTATION — F36: validación relajada de fries + jerárquico con DINOv2-L

## FASE 36 — DINOv2 Rice Validation + Hierarchical ✅ (2026-08-31)
- [x] Validación RELAXED: fries 19 VALID (16 strict + 3 relaxed) — bloqueo resuelto
- [x] RICE con fries ampliadas + DINO: 44% ± 22.7 (F35 48.8) — SIN mejora: el 48.8% NO era solo datos; el grupo es resistente al DINO frozen (el inter 0.218 no se traduce en clasificación)
- [x] Híbrido legacy + DINO pizza/naan (100% vs 88.3): global estimado 74.1% > 71.3 ✓ — la vía viable
- [x] DECISIÓN: APPROVE EXPERIMENTAL (híbrido); rice pendiente de otra estrategia (auditar GT fries + fine-tuning candidato)

## FASE 37 — Specialist Router + Runtime Validation ✅ (2026-08-31)
- [x] SpecialistRouter: legacy + DINO pizza/naan (3 seeds) con gates A/B/C
- [x] RUNTIME CONFIRMADO: Gate B (conf<0.40 + top-3) = 74.1% (80/108) — el 74.1% estimado de F36 MEDIDO; pizza 20/20, 21 llamadas DINO (19.4%), 0 falsos, 0 perdidas; Gate A no rescata (71.3)
- [x] Latencia: +887ms por llamada (19.4% de imágenes) = +172ms promedio — aceptable para el specialist selectivo
- [x] DECISIÓN: APPROVE EXPERIMENTAL — rollout controlado (legacy_specialists), no default; rice NO añadir (evidencia negativa)

## FASE 38 — Controlled Rollout + Shadow + DINO-base ✅ (2026-08-31)
- [x] Threshold sweep 0.20-0.50: 0.35+ estable 74.1%, 0 regresiones en todo el rango; 0.40 confirmado
- [x] Shadow telemetría: 19.4% invocación, 3 correcciones, 0 regresiones, 1 abstención — offline == runtime (74.1 reproducible)
- [x] DINO-base == DINO-L en accuracy (74.1) con 264ms vs 902ms -> BASE preferible (Caso D)
- [x] SpecialistRouter como abstracción (app/models/specialist_router.py) + 6 tests unitarios (78/78 totales)
- [x] DECISIÓN: APPROVE EXPERIMENTAL -> rollout controlado preparado (env config, rollback garantizado, backward compat testado)

## FASE 39 — Shadow en tráfico real ✅ parcial (2026-08-31)
- [x] SpecialistShadow module + flags + integración analyze/main/config (78/78 tests; el resultado legacy NUNCA cambia — verificado)
- [x] Flujo directo verificado: pizza_001 -> invoked, specialist=pizza, would_change=False, final=pizza
- [x] Shadow offline reproducido (21 calls, 3 corr, 0 regr, 1 abst, 74.1% potencial)
- [ ] RUNTIME: el specialist NO se activó en el uvicorn (Settings lee la env en proceso directo pero no en el Start-Process — causa pendiente de depurar) — REQUIERE CORRECCIÓN antes de shadow real

## FASE 40 — Activación runtime + aprobación shadow ✅ (2026-08-31)
- [x] CAUSA RAÍZ de F39: el logger escribe a STDERR — el F39 buscaba en STDOUT (falso negativo); el shadow SIEMPRE estuvo activado
- [x] VERIFICADO en uvicorn real: SPECIALIST CONFIG enabled=True threshold=0.4 groups=pizza,naan model=dino_base available=True
- [x] Invariancia: el hook del shadow va DESPUÉS de construir el response (solo loguea) — pizza_001 idéntica con shadow ON/OFF
- [x] DECISIÓN: READY FOR REAL SHADOW (validación food-us por API pendiente por inestabilidad de Start-Process local, no del código)

## FASE 41 — Open Food Retrieval Engine ✅ parcial / REQUIRES CORRECTION (2026-08-31)
- [x] Diagnóstico del API FNDDS: query obligatorio, %20 en espacios, pageSize=50 (100/25 -> 400), load_dotenv override — los 4 bugs resueltos en código
- [ ] BLOQUEO: el archivo del importador fue eliminado del disco externamente (patrón de la otra sesión del monorepo) — el catálogo 1.000+ no completó
- [ ] DECISIÓN: REQUIRES CORRECTION — reintento mecánico con los fixes documentados

## FASE 42 — Completar F41: catálogo 1.451 + embeddings + retrieval ✅ (2026-08-31)
- [x] Importador recuperado del commit 8d3691b + 2 bugs nuevos resueltos (pageNumber 1-indexed, retry intermitente)
- [x] Catálogo: 1.451 entradas (FNDDS CC0) con canonical genérico + aliases + nutrition_mapping
- [x] Embeddings CLIP 3 templates cacheados (1451x512) + índice numpy exacto + motor FoodRetrieval
- [x] R@K: food-us 38.9/51.9/57.4/58.3/64.8; v1 25/34.3/38.8/40.7/45.5; Food-101 31.5/42.2/46.3/49.5/52.7 — el correcto aparece en Top-K con catálogo 1.451 sin reentrenar
- [x] Flags retrieval off (producción intacta); 75/75 tests; smoke: pizza_001 -> top-5 pizza

## FASE 43 — Canonical Grouping + Reranking ✅ (2026-08-31)
- [x] Canonical grouping: R@K mejorado (food-us R@5 51.9 -> 58.3, R@10 57.4 -> 63.0); unique10 6.0 (variantes agrupadas)
- [x] Rerank con legacy (0.75+0.25): SIN cambio (correlación CLIP — el legacy y el retrieval comparten el embedding)
- [x] Cuello: retrieval textual (el mismo del F26); el R@1 38.9% requiere el specialist DINO selectivo (F38: 74.1% con hybrid)
- [x] DECISIÓN: grouping aprobado; el rerank estructural = integrar el specialist DINO al ranking del retrieval

## FASE 45 — Calibración final + congelación ✅ (2026-08-31)
- [x] Sweep 3x3 (gate top-3/5/10 x th 0.65/0.70/0.75): config final Top-3 + 0.75 — regr 42 -> 22 (-48%), R@1 41.7 conservado, F101 -0.2
- [x] Regresiones explícitas: 17, TODAS lasagna/nachos -> pizza (conf 0.75-0.82) — categoría única
- [x] Config congelada: SPECIALIST_MODEL=dino_base GROUPS=pizza,naan THRESHOLD=0.75 GATE_TOPK=3 GATE_CONF=0.40 ENABLED=false
- [x] app/models/retrieval_rerank.py (regla pura) + 9 tests (84/84)
- [x] NOTA: SpecialistRouter/Shadow en feature/f38 (rama ajena) — traer para F46

## FASE 46 — Shadow Real del Retrieval + Specialist ✅ (2026-09-03)
- [x] SpecialistRouter/Shadow recuperados de feature/f38 (la rama carlos no los tenía)
- [x] RetrievalShadow (retrieval+grouping+rerank) + startup log RETRIEVAL CONFIG + flag retrieval_shadow_enabled=false
- [x] food-us por API real: 108 peticiones, telemetría 48 líneas: canonical hit@1 33.3%, reranked 34.3%, spec 14.8%, fallback 0, lat p50 98ms
- [x] Invariancia: pizza_001 OFF vs ON -> pizza 0.9075 idéntico (hook post-response)
- [x] Concurrencia 4 simultáneos: 200x4, 0 crashes; fallbacks verificados (DINO unavailable + retrieval error + DetectorBased real)
- [x] 5 tests shadow (89/89) — DECISIÓN: APPROVE F46 READY TO SCALE CATALOG

## FASE 47 — Escalado del catálogo (4x, sin reentrenar) ⚠️ PARCIAL (2026-09-04)
- [x] Catálogo 1.451 -> 5.761 canónicos (9.067 entries, 1.737 aliases): FNDDS completo + SR Legacy (list API paginado) + OFF (cache incremental por query; OFF inestable 503/401)
- [x] 1.451 preservados (foods_legacy_1451.json + cross-check legacy + naan restaurado como alias)
- [x] Embeddings 5.761x512 x3 + índice numpy (build ~5s, RAM ~12MB, matmul ~2ms)
- [x] R@K degradación esperada: food-us 38.9 -> 10.2 (espacio 4x; documentado para F48 — NO reducir catálogo)
- [x] 8 tests (97/97): count, unicidad, aliases, cobertura, rebuild, backcompat, smoke, fuentes
- [ ] 10k estricto NO alcanzado: política no-variantes limita a ~5.8k reales — nivel variante u OFF estable = decisión F48

## FASE 48 — General Reranker para 5.761 ✅ APPROVE (2026-09-04)
- [x] Reranker determinista (retrieval + support + alias_count + specialist DINO): food-us 10.2 -> 38.9 (recupera el nivel del 1.451), v1 8.7 -> 21.0, Food-101 9.5 -> 34.2 (superado) — sin entrenar
- [x] Alias_count (proxy del canónico genérico) = señal clave; support aporta poco; specialist +0.9 (regla F45 intacta)
- [x] MRR 0.185 -> 0.458, med_rank 6 -> 1, prom 43/dem 4 (food-us); buckets: A=46% (GT fuera del Top-50 — recall, no ranking)
- [x] app/models/retrieval_rerank.py: rerank_general (puro) + 7 tests (104/104)
- [x] DECISIÓN: APPROVE — el ranking escala con el catálogo; el cuello restante = recall del retrieval (bucket A)

## FASE 49 — Recall Expansion ⚠️ REQUIRES BETTER REPRESENTATION (2026-09-04)
- [x] K sweep: R@200 food-us 73.1 (+19.4), v1 45.6, food101 60.1; bucket A 46 -> 27%
- [x] F49-A top-200: food-us 43.5 (+4.6) pero v1 16.5 / food101 19.9 (REGRESIÓN — el reranker F48 calibrado para top-50 no escala al pool-200)
- [x] F49-B multi-query (4 vistas): sin ganancia (vistas del mismo CLIP correlacionan); el multi no supera el single
- [x] A4 (>500): 26-44% GT fuera — el límite = representación CLIP textual, no el pool
- [x] fuse_views en retrieval_rerank + 6 tests (110/110)
- [x] DECISIÓN: REQUIRES BETTER RETRIEVAL REPRESENTATION — NO escalar a 10k; siguiente: aliases como prompts del índice (sin entrenar)

## FASE 50 — Multi-Text Retrieval (canonical + aliases) ✅ APPROVE (2026-09-04)
- [x] Índice 7.498 textos (canonical + aliases) x3 templates = 22.494 embeddings (15MB RAM, +25%)
- [x] Recall: food-us R@500 74.1 -> 90.7; bucket A500 26% -> 9%; R@50 +22; 356 rescates por alias (ejemplos: french fries <- potato french fries school)
- [x] Aggregación max > top-2mean; R@1 retrieval solo: 38.0 food-us (= F48 completo)
- [x] group_text_matches en retrieval_rerank + 7 tests (117/117)
- [x] DECISIÓN: APPROVE — el recall soporta el F51 (segmentación + reranker) y el F52 (10k)

## FASE 51 — Integración Final del Pipeline 5.761 ✅ APPROVE (2026-09-04)
- [x] app/models/food_pipeline.py: multi-text retrieval -> grouping -> reranker F48 -> specialist DINO (F45) -> canonical; fallback seguro
- [x] Fix calibración: support=1+aliases, alias=min(20) — el reranker F48 (calibrado 1.451) degradaba con el v4 (31.5 -> 56.5 food-us)
- [x] Integración API: flags retrieval_enabled/shadow_enabled (default false, legacy exact); shadow telemetría pipeline_f51
- [x] R@1 integrado: food-us 56.5 (vs F50 38.0), v1 21.8, food101 27.1; invariancia pizza 0.9075 off==on; concurrencia 4x200; smoke ✓
- [x] 7 tests pipeline (124/124) — DECISIÓN: APPROVE READY FOR NUTRITION

## FASE 52 — Nutrition Mapping Masivo ⚠️ PARTIAL (2026-09-04)
- [x] 96% de los 5.761 con fdc_id reales (USDA FNDDS/SR + OFF); 1.184 con nutrientes hoy (límite diario FDC 1000 — reanudable, 0 errores)
- [x] NutritionService: lookup local (cero red), per-100g, unidades explícitas, NUTRITION_READY/UNAVAILABLE, confianza por fuente (0.95 USDA / 0.85 OFF)
- [x] Golden foods validados (pizza 292, rice 119, fries 185 kcal/100g); muestra 50: 38 correct/7 acceptable/5 unavailable/0 ambiguous
- [x] Flags nutrition_enabled/shadow_enabled=false + shadow hook + 10 tests (134/134)
- [x] DECISIÓN: PARTIAL SUCCESS — el límite diario del USDA pospone el 100% (~4 días de tandas); pipeline funcional

## FASE 53 — Confidence + Fallback ✅ APPROVE (2026-09-04)
- [x] DecisionPolicy: NEW_RESULT_READY/NUTRITION_UNAVAILABLE/LOW_CONFIDENCE/LEGACY_FALLBACK; identificación vs nutrición INDEPENDIENTES (no revierte identificación por falta de nutrientes)
- [x] Confianzas separadas (visual 0.298 runtime / nutrition 0.95 USDA / portion externa); flags confidence_enabled=false + min_visual_confidence=0.20
- [x] Shadow de decisión runtime: pizza -> NEW_RESULT_READY + nutrición real (292 kcal); golden cases A-E en tests
- [x] 8 tests (142/142) — DECISIÓN: APPROVE READY FOR FINAL SHADOW

## FASE 54 — Nutrition Coverage + Shadow Final ✅ APPROVE (2026-09-04)
- [x] Nutrition: 1.184 -> 2.184 (37.9%, USDA 2.000 + OFF 184, 0 errores, 0 NaN/negativos, tandas reanudables)
- [x] Shadow final food-us (107 por API): new 56.1% vs legacy 43.9% (+12.2); A=44 B=3 C=16 D=44; nutrition READY 91.6% runtime
- [x] Invariancia OFF==ON (pizza 0.9075); concurrencia 4x200; muestra 50: 41 correct/9 acceptable
- [x] DECISIÓN: APPROVE READY FOR ACTIVE ROLLOUT

## FASE 55 — Active Rollout Controlado ✅ APPROVE (2026-09-04)
- [x] Fix crop del active (CLIP del pipeline, no DetectorBased): 38.9 -> 47.7% bruto / 70.8% sobre detectados (> legacy 65.3%)
- [x] Confidence del active = retrieval score real (no 0.5 fijo); fallback/rollback por flag probados (legacy exacto)
- [x] 108 requests controlados, 0 crashes, contract intacto, nutrition 37.9% (tandas independientes)
- [x] 6 tests (148/148) — DECISIÓN: APPROVE READY FOR GRADUAL PRODUCTION

## FASE 56 — Rollout Gradual Real ✅ APPROVE (2026-09-04)
- [x] 5 etapas (0/10/25/50/100%): el nuevo supera al legacy en TODAS (10%: 70.0 vs 43.3; 100%: 47.7% = 70.8% detectados); 0 fallos del pipeline; 1 error = imagen corrupta del dataset
- [x] Latencia estable p50 ~2.2-2.4s; rollback por flag (etapa 0% = rollback ejecutado); nutrition 37.9% independiente
- [x] scripts/f56_rollout.py (split determinista por hash) + 4 tests (152/152)
- [x] DECISIÓN: APPROVE FULL PRODUCTION ROLLOUT — siguiente: F57 hardening + monitoring

## FASE 57 — Hardening + Monitoring ✅ APPROVE (2026-09-04)
- [x] RELEASE versionado (f57/5761/multitext-v1/reranker-v1/specialist-v1/nutrition-v1) en startup
- [x] /health/readiness (runtime: ready, index/clip/dino/nutrition True) + /metrics (contadores por request, runtime verificado)
- [x] Consistencia catálogo 5761 == índice 7498; fallback reasons separados; rollback por flag documentado; nutrition 37.9% observable
- [x] 8 tests (160/160) — DECISIÓN: APPROVE PRODUCTION HARDENED — sistema operable, observable, reversible
