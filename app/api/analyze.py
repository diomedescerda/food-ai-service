import os
import time
from io import BytesIO
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from app.models.base import IFoodDetector
from app.models.detection import Detection
from app.schemas.analyze import (
    AnalyzeResponse,
    BoundingBox,
    DetectedFood,
    PortionEstimate,
    Segmentation,
)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024

# Firmas mágicas: JPEG (FF D8 FF), PNG (89 50 4E 47), WEBP ("RIFF" .... "WEBP")
MAGIC_BYTES = {
    "image/jpeg": b"\xff\xd8\xff",
    "image/png": b"\x89\x50\x4e\x47",
    "image/webp": b"RIFF",
}


def _error(code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={"success": False, "error": {"code": code, "message": message}},
    )


def _has_valid_magic(content_type: str, head: bytes) -> bool:
    expected = MAGIC_BYTES.get(content_type)
    if expected is None:
        return False
    return head.startswith(expected)


def _get_detector(request: Request) -> IFoodDetector:
    detector = request.app.state.detector
    if detector is None or not detector.is_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": {"code": "MODEL_NOT_READY", "message": "El modelo no está cargado."},
            },
        )
    return detector


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analyze"])
async def analyze(
    request: Request,
    analysis_id: str = Form(...),
    image: UploadFile = File(...),
) -> AnalyzeResponse:
    """Detecta alimentos en la imagen: clase + confidence + bounding box (píxeles)."""
    try:
        UUID(analysis_id)
    except ValueError:
        raise _error("INVALID_ANALYSIS_ID", "analysis_id debe ser un UUID válido.")

    if image.size is None or image.size <= 0:
        raise _error("EMPTY_FILE", "El archivo está vacío.")

    if image.size > MAX_IMAGE_SIZE_BYTES:
        raise _error("IMAGE_TOO_LARGE", "La imagen supera el tamaño máximo.")

    content_type = (image.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise _error("INVALID_IMAGE", f"Tipo de contenido no permitido: {content_type or 'desconocido'}.")

    filename = image.filename or ""
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise _error("INVALID_IMAGE", f"Extensión no permitida: {extension}.")

    head = await image.read(12)
    await image.seek(0)
    if not _has_valid_magic(content_type, head):
        raise _error("CORRUPT_FILE", "El archivo no es una imagen válida.")

    try:
        pil_image = Image.open(BytesIO(await image.read())).convert("RGB")
    except Exception:
        raise _error("CORRUPT_FILE", "No fue posible decodificar la imagen.")

    detector = _get_detector(request)
    segmenter = getattr(request.app.state, "segmenter", None)
    classifier = getattr(request.app.state, "classifier", None)

    # Serializa la inferencia CPU (FASE 23): torch en CPU NO soporta DINO
    # concurrente (crash nativo sin traceback bajo >=4 requests simultáneos).
    # Los requests se aceptan pero el pipeline se ejecuta de uno en uno.
    semaphore = getattr(request.app.state, "inference_semaphore", None)
    if semaphore is not None:
        await semaphore.acquire()
    try:
        return await _run_pipeline(request, detector, segmenter, classifier, pil_image,
                                   analysis_id, image.content_type or "", image.filename or "")
    finally:
        if semaphore is not None:
            semaphore.release()


async def _run_pipeline(
    request: Request,
    detector: IFoodDetector,
    segmenter,
    classifier,
    pil_image: Image.Image,
    analysis_id: str,
    content_type: str,
    filename: str,
) -> AnalyzeResponse:
    start = time.perf_counter()
    detections = detector.detect(pil_image)
    det_ms = round((time.perf_counter() - start) * 1000)
    used_dino = bool(getattr(detector, "used_dino_fallback", False))

    seg_start = time.perf_counter()
    segmentations: list = [None] * len(detections)
    if segmenter is not None and segmenter.is_loaded and detections:
        segmentations = segmenter.segment(pil_image, detections)
    seg_ms = round((time.perf_counter() - seg_start) * 1000)

    # Clasificación final: con DetectorBasedClassifier no hay inferencia extra
    # (la clase/confianza vienen del detector); el pipeline habla contra
    # IFoodClassifier para poder sustituirlo por un clasificador dedicado.
    # FASE 51: si FOOD_AI_RETRIEVAL_ENABLED=true, el pipeline 5.761 sustituye
    # la clase por el canonical del multi-text retrieval (experimental).
    pipeline = getattr(request.app.state, "food_pipeline", None)
    pipeline_on = (
        pipeline is not None and pipeline.available()
        and getattr(request.app.state.settings, "retrieval_enabled", False)
    )
    if pipeline_on:
        cls_start = time.perf_counter()
        final_names = []
        final_confidences = []
        for d in detections:
            crop = classifier._crop(pil_image, d) if classifier is not None and hasattr(classifier, "_crop") else None
            if crop is None:
                crop = pil_image
            res = pipeline.analyze_food(crop)
            if res.get("fallback", False) or not res.get("canonical_name"):
                final_names.append(d.name)
                final_confidences.append(d.confidence)
            else:
                final_names.append(res["canonical_name"])
                final_confidences.append(0.5)
        cls_ms = round((time.perf_counter() - cls_start) * 1000)
    elif classifier is not None and classifier.is_loaded:
        cls_start = time.perf_counter()
        classifications = classifier.classify(pil_image, detections)
        cls_ms = round((time.perf_counter() - cls_start) * 1000)
        final_names = [c.name if c is not None else d.name for c, d in zip(classifications, detections)]
        final_confidences = [c.confidence if c is not None else d.confidence for c, d in zip(classifications, detections)]
    else:
        final_names = [d.name for d in detections]
        final_confidences = [d.confidence for d in detections]
        classifier = None
        cls_ms = 0

    # F51: scores legacy para el gate del specialist (solo si el pipeline está
    # activo; el pipeline legacy no cambia). Usa el CLIP propio del pipeline.
    legacy_conf1 = 0.0
    legacy_top3: list[str] = []
    shadow_crop = None
    pipeline_active = getattr(request.app.state, "food_pipeline", None)
    pipeline_clf = getattr(pipeline_active, "clf", None) if pipeline_active is not None else None
    if (
        pipeline_active is not None and pipeline_active.available()
        and pipeline_clf is not None and detections
        and hasattr(pipeline_clf, "_score_crop") and hasattr(pipeline_clf, "_crop")
    ):
        try:
            shadow_crop = pipeline_clf._crop(pil_image, detections[0])
            if shadow_crop is not None:
                _best: dict[str, float] = {}
                for r in pipeline_clf._score_crop(shadow_crop):
                    _best[r.name] = max(_best.get(r.name, 0.0), r.score)
                legacy_conf1 = max(_best.values()) if _best else 0.0
                legacy_top3 = [n for n, _ in sorted(_best.items(), key=lambda x: -x[1])[:3]]
        except Exception:  # noqa: BLE001 — el pipeline jamás rompe la respuesta
            legacy_conf1 = 0.0
            legacy_top3 = []

    # Estimación de porción (básica): referencia + tamaño visual relativo.
    # estimatedGrams != measuredGrams — aproximación, nunca peso medido.
    # La porción usa la IDENTIDAD CLASIFICADA (CLIP), no el nombre del
    # detector ("unknown" en fallback DINO) — FASE 16.
    portion_estimator = getattr(request.app.state, "portion_estimator", None)
    portion_start = time.perf_counter()
    portions: list = [None] * len(detections)
    if portion_estimator is not None and detections:
        classified_detections = [
            Detection(name=name, confidence=conf, bounding_box=d.bounding_box)
            for d, name, conf in zip(detections, final_names, final_confidences)
        ]
        portions = portion_estimator.estimate(pil_image, classified_detections, segmentations)
    portion_ms = round((time.perf_counter() - portion_start) * 1000)

    # Debug visual: overlay + depth map (herramienta de desarrollo).
    from app.utils.debug import save_debug_image, save_depth_image

    debug_path = save_debug_image(
        request.app.state.settings.debug_images_dir,
        analysis_id,
        pil_image,
        detections,
        segmentations,
    )
    if debug_path:
        request.app.state.logger.debug("Debug image guardada: %s", debug_path)

    inference_ms = det_ms + seg_ms + portion_ms + cls_ms

    # Observabilidad (FASE 22): telemetría por análisis — conteos por etapa,
    # fallback DINO y tiempos. Sin datos de usuario ni imágenes.
    foods_unknown = sum(1 for n in final_names if n == "unknown")
    if request.app.state.logger:
        request.app.state.logger.info(
            "análisis_completo analysis_id=%s foods_detected=%d foods_classified=%d "
            "foods_unknown=%d food_instance_count=%d used_dino_fallback=%s "
            "detector_ms=%d segmentation_ms=%d classification_ms=%d portion_ms=%d "
            "total_ms=%d status=%s",
            analysis_id, len(detections), len(final_names) - foods_unknown, foods_unknown,
            len(final_names), used_dino, det_ms, seg_ms, cls_ms, portion_ms,
            inference_ms, "completed",
        )

    depth_estimator = getattr(request.app.state, "depth_estimator", None)
    if depth_estimator is not None and depth_estimator.is_loaded:
        depth_start = time.perf_counter()
        depth_map = depth_estimator.estimate(pil_image)
        depth_ms = round((time.perf_counter() - depth_start) * 1000)
        request.app.state.logger.debug(
            "Depth map: %sx%s, stats=%s (%d ms)",
            depth_map.width, depth_map.height, depth_map.stats(), depth_ms,
        )
        save_depth_image(
            request.app.state.settings.debug_images_dir,
            analysis_id,
            depth_map,
        )

# F51: shadow del pipeline 5.761 (telemetría; el resultado SIEMPRE es legacy).
    # Se ejecuta DESPUÉS de construir el response — jamás puede modificarlo.
    pipeline = getattr(request.app.state, "food_pipeline", None)
    if (
        pipeline is not None and pipeline.available() and request.app.state.logger
        and getattr(request.app.state.settings, "retrieval_shadow_enabled", False)
    ):
        try:
            crop = shadow_crop if shadow_crop is not None else pil_image
            tel = pipeline.analyze_food(crop, legacy_conf1=legacy_conf1, legacy_top3=legacy_top3)
            if not tel.get("fallback", False):
                request.app.state.logger.info(
                    "pipeline_f51 analysis_id=%s retrieval1=%s reranked1=%s final=%s "
                    "spec_used=%s spec_conf=%s lat=%sms embed=%sms fallback=%s",
                    tel.get("retrieval", {}).get("top1"), tel.get("reranker", {}).get("top1"),
                    tel.get("canonical_name"), tel.get("specialist_used", False),
                    tel.get("specialist", {}).get("confidence"), tel.get("latency_ms"),
                    tel.get("embedding_ms"), tel.get("fallback", False),
                )
        except Exception as exc:  # noqa: BLE001 — el pipeline jamás rompe la respuesta
            request.app.state.logger.error("pipeline F51 shadow error (ignorado): %s", exc)

    return AnalyzeResponse(
        analysis_id=analysis_id,
        status="completed",
        model_version=detector.model_version,
        seg_model_version=(
            segmenter.model_version if segmenter is not None and segmenter.is_loaded else "none"
        ),
        classifier_version=(
            classifier.model_version if classifier is not None and classifier.is_loaded else "none"
        ),
        inference_time_ms=inference_ms + seg_ms + portion_ms,
        foods=[
            DetectedFood(
                name=name,
                confidence=confidence,
                bounding_box=BoundingBox(
                    x=d.bounding_box.x,
                    y=d.bounding_box.y,
                    width=d.bounding_box.width,
                    height=d.bounding_box.height,
                ),
                segmentation=(
                    Segmentation(mask=s.mask, area_pixels=s.area_pixels)
                    if s is not None
                    else None
                ),
                portion=(
                    PortionEstimate(
                        portion_size=p.portion_size,
                        estimated_grams=p.estimated_grams,
                        min_grams=p.min_grams,
                        max_grams=p.max_grams,
                        confidence=p.confidence,
                        method=p.method,
                    )
                    if p is not None
                    else None
                ),
            )
            for d, s, p, name, confidence in zip(detections, segmentations, portions, final_names, final_confidences)
        ],
    )