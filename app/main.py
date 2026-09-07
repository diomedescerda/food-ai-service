import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.analyze import router as analyze_router
from app.api.health import router as health_router
from app.core.config import settings
from app.models.advanced_portion_estimator import AdvancedPortionEstimator
from app.models.base import IFoodDetector
from app.models.basic_portion_estimator import BasicPortionEstimator
from app.models.depth_anything_estimator import DepthAnythingEstimator
from app.models.detector_based_classifier import DetectorBasedClassifier
from app.models.hybrid_detector import GroundingDinoDetector, HybridFoodDetector
from app.models.yolo_food_detector import YoloFoodDetector
from app.models.yolo_food_segmenter import YoloFoodSegmenter
from app.models.zero_shot_classifier import ZeroShotFoodClassifier
from app.services.portion_geometry import PortionGeometryEstimator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida: carga detector, segmentador, clasificador y estimadores
    UNA vez al arrancar (nunca por request)."""
    app.state.logger = logging.getLogger("uvicorn.error")
    detector: IFoodDetector
    if settings.detector_type == "hybrid":
        detector = HybridFoodDetector(
            yolo=YoloFoodDetector(settings),
            dino=GroundingDinoDetector(settings.dino_model, settings.dino_prompt, settings.dino_threshold),
        )
        app.state.logger.info("Detector híbrido: YOLO + DINO (fallback open-vocabulary)")
    elif settings.detector_type == "dino":
        detector = GroundingDinoDetector(settings.dino_model, settings.dino_prompt, settings.dino_threshold)
        app.state.logger.info("Detector DINO (open-vocabulary)")
    else:
        detector = YoloFoodDetector(settings)
    detector.load()
    app.state.detector = detector
    app.state.logger.info(
        "Food detector cargado: version=%s, path=%s, classes=%s",
        detector.model_version,
        settings.model_path,
        sorted(detector.supported_classes),
    )

    segmenter = YoloFoodSegmenter(settings)
    segmenter.load()
    app.state.segmenter = segmenter
    app.state.logger.info(
        "Food segmenter cargado: version=%s, path=%s",
        segmenter.model_version,
        settings.seg_model_path,
    )

    # Clasificador: detector_based (clase de YOLO) o zero_shot (CLIP, catálogo
    # amplio). A/B configurable con FOOD_AI_CLASSIFIER_TYPE.
    if settings.classifier_type == "zero_shot":
        ensemble = tuple(
            t.strip() for t in settings.clip_prompt_ensemble.split("|") if t.strip()
        )
        classifier = ZeroShotFoodClassifier(
            model_name=settings.clip_model,
            device=settings.clip_device,
            threshold=settings.clip_threshold,
            prompt_template=settings.clip_prompt_template,
            prompt_templates_extra=ensemble,
            crop_padding=settings.clip_crop_padding,
        )
        classifier.load()
        app.state.logger.info(
            "Clasificador zero-shot cargado: version=%s, threshold=%s, catálogo=%d candidatos",
            classifier.model_version, settings.clip_threshold, len(classifier._candidates),
        )
    else:
        classifier = DetectorBasedClassifier(detector)
        app.state.logger.info("Clasificador cargado: version=%s", classifier.model_version)
    app.state.classifier = classifier

    # Estimador de porción: basic (referencia) o advanced (depth).
    app.state.depth_estimator = None
    if settings.portion_method == "advanced":
        if settings.depth_enabled:
            depth = DepthAnythingEstimator(settings.depth_model_path, settings.depth_device)
            depth.load()
            app.state.depth_estimator = depth
            app.state.logger.info(
                "Estimador de profundidad cargado: version=%s (carga única)",
                depth.model_version,
            )
        app.state.portion_estimator = AdvancedPortionEstimator(app.state.depth_estimator)
        app.state.logger.info("Estimador de porción cargado: advanced_depth_relative")
    else:
        app.state.portion_estimator = BasicPortionEstimator()
        app.state.logger.info("Estimador de porción cargado: basic_reference")

    app.state.settings = settings
    # FASE 23: serializa la inferencia CPU (torch concurrente con DINO crashea).
    app.state.inference_semaphore = asyncio.Semaphore(1)

    # FASE 51: pipeline único 5.761 alimentos (retrieval multi-text +
    # grouping + reranker F48 + specialist DINO). Flag: retrieval_enabled
    # (respuesta = pipeline) o retrieval_shadow_enabled (telemetría sin
    # tocar la respuesta). Default false -> legacy exacto.
    app.state.food_pipeline = None
    app.state.logger.info(
        "RETRIEVAL CONFIG: enabled=%s shadow=%s catalog_size=%s specialist_model=dino_base "
        "specialist_threshold=0.75 specialist_gate_topk=3 specialist_groups=pizza,naan",
        settings.retrieval_enabled, settings.retrieval_shadow_enabled,
        "5761" if settings.retrieval_enabled or settings.retrieval_shadow_enabled else "n/a",
    )
    if settings.retrieval_enabled or settings.retrieval_shadow_enabled:
        from app.models.food_pipeline import FoodPipeline  # noqa: PLC0415
        from app.models.zero_shot_classifier import ZeroShotFoodClassifier  # noqa: PLC0415

        if not isinstance(classifier, ZeroShotFoodClassifier):
            clip = ZeroShotFoodClassifier(
                model_name=settings.clip_model,
                device=settings.clip_device,
                threshold=settings.clip_threshold,
                prompt_template=settings.clip_prompt_template,
                crop_padding=settings.clip_crop_padding,
            )
            clip.load()
            pipeline_clf = clip
        else:
            pipeline_clf = classifier
        app.state.food_pipeline = FoodPipeline(clf=pipeline_clf, enabled=True)
        app.state.logger.info(
            "pipeline F51 listo: available=%s índice=%s",
            app.state.food_pipeline.available(),
            app.state.food_pipeline.index.shape if app.state.food_pipeline.index is not None else None,
        )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description="Servicio de IA de FoodAI: detección, segmentación, clasificación, porción y nutrición.",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(analyze_router)
    return app


app = create_app()