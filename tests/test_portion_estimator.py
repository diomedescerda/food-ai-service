"""Tests del estimador de porción básico (sin modelos reales).

Cubren: small/medium/large/unknown, min <= estimated <= max, confidence 0..1,
sin máscara (fallback bbox), alimento sin referencia y sin división por cero.
"""

from PIL import Image

from app.models.basic_portion_estimator import BasicPortionEstimator
from app.models.detection import BoundingBox, Detection
from app.models.segmenter_base import SegmentationResult

ESTIMATOR = BasicPortionEstimator()
IMAGE_1000 = Image.new("RGB", (1000, 1000), (0, 0, 0))


def _segmentation(area: int) -> SegmentationResult:
    return SegmentationResult(mask="b64", area_pixels=area)


def _detection(name: str, box: BoundingBox) -> Detection:
    return Detection(name=name, confidence=0.9, bounding_box=box)


def _estimate(name: str, mask_area: int, image: Image.Image = IMAGE_1000):
    detections = [_detection(name, BoundingBox(100, 100, 800, 800))]
    segmentations = [_segmentation(mask_area)]
    return ESTIMATOR.estimate(image, detections, segmentations)[0]


def test_small():
    est = _estimate("banana", 10_000)  # 1% del área → small
    assert est.portion_size == "small"
    assert est.estimated_grams < 118
    assert est.min_grams <= est.estimated_grams <= est.max_grams


def test_medium():
    est = _estimate("banana", 200_000)  # 20% → medium
    assert est.portion_size == "medium"
    assert est.estimated_grams == 118  # referencia FDC banana mediana


def test_large():
    est = _estimate("banana", 400_000)  # 40% → large
    assert est.portion_size == "large"
    assert est.estimated_grams > 118


def test_unknown_sin_referencia():
    est = _estimate("sandwich", 200_000)
    assert est.portion_size == "unknown"
    assert est.estimated_grams is None
    assert est.min_grams is None
    assert est.max_grams is None
    assert est.confidence == 0.0
    assert est.method == "unavailable"


def test_rango_min_estimated_max():
    for name in ("banana", "apple", "orange", "pizza", "hot dog", "donut", "cake", "carrot", "broccoli"):
        for area in (10_000, 200_000, 400_000):
            est = _estimate(name, area)
            assert est.min_grams is not None
            assert est.max_grams is not None
            assert est.estimated_grams is not None
            assert est.min_grams <= est.estimated_grams <= est.max_grams, (
                f"{name}/{est.portion_size}: {est.min_grams} <= {est.estimated_grams} <= {est.max_grams}"
            )


def test_confidence_entre_0_y_1():
    est = _estimate("banana", 200_000)
    assert 0.0 <= est.confidence <= 1.0
    assert est.confidence == 0.55


def test_sin_mascara_fallback_bbox():
    detections = [_detection("banana", BoundingBox(0, 0, 300, 300))]  # 9% → small
    estimates = ESTIMATOR.estimate(IMAGE_1000, detections, [None])
    assert estimates[0].portion_size == "small"


def test_imagen_de_area_cero_no_divide_por_cero():
    detections = [_detection("banana", BoundingBox(0, 0, 10, 10))]
    estimates = ESTIMATOR.estimate(Image.new("RGB", (0, 0)), detections, [None])
    assert estimates[0].portion_size == "unknown"
    assert estimates[0].confidence == 0.0


def test_sin_detecciones():
    estimates = ESTIMATOR.estimate(IMAGE_1000, [], [])
    assert estimates == []


def test_porcion_es_estimacion_no_medicion():
    est = _estimate("banana", 200_000)
    assert est.method == "basic_reference"
    assert est.estimated_grams is not None
    # estimatedGrams != measuredGrams: documentado en la clase y docs.


def test_normalizacion_alias_guion_a_espacio():
    """FASE 16: CLIP devuelve canonical con guiones (hot_dog); REFERENCE_GRAMS usa espacios."""
    detections = [_detection("hot_dog", BoundingBox(100, 100, 200, 200))]
    estimates = ESTIMATOR.estimate(IMAGE_1000, detections, [_segmentation(200_000)])
    assert estimates[0].portion_size == "medium"
    assert estimates[0].estimated_grams == 57  # FDC: 1 frankfurter