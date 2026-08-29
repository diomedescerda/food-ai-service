"""Estimador de porción básico — explícitamente aproximado.

Algoritmo simple y explicable (sin depth, sin volumen, sin escala física):

1. Referencia por alimento: gramos de UNA porción doméstica típica según USDA
   FoodData Central (medidas domésticas, docs/portion-estimation.md). Si el
   alimento no tiene referencia → UNKNOWN (nunca se inventa un número).
2. Tamaño visual: área relativa de la máscara (mask_area / image_area) con
   umbrales fijos y documentados. Fallback: área del bbox si no hay máscara.
   NOTA: área en píxeles NO equivale a gramos (sin escala física); solo
   discrimina small/medium/large.
3. Gramos: estimated = base × factor(tamaño); rango [min, max] alrededor.
4. Confidence fija 0.55 (heurística visual simple, documentada); UNKNOWN → 0.

Factores (±20 % sobre la referencia) documentados en docs/portion-estimation.md.
"""

from PIL import Image

from app.models.detection import Detection
from app.models.portion_base import IPortionEstimator, PortionEstimate
from app.models.segmenter_base import SegmentationResult

# Referencias: gramos de UNA porción doméstica típica (USDA FDC).
# Alias = nombre de clase YOLO (igual que los aliases de nutrición).
# Referencias: gramos de UNA porción doméstica típica (USDA FDC — SR Legacy/FNDDS).
# Verificadas 2026-08-28 vía registros FDC (myfooddata espejo de USDA, mismos IDs).
REFERENCE_GRAMS: dict[str, int] = {
    "banana": 118,       # FDC: 1 banana mediana (7"-7⅞") ≈ 118 g
    "apple": 182,        # FDC: 1 manzana mediana ≈ 182 g
    "orange": 131,       # FDC: 1 naranja mediana ≈ 131 g
    "broccoli": 91,      # FDC: 1 taza de brócoli crudo picado ≈ 91 g
    "carrot": 61,        # FDC: 1 zanahoria mediana ≈ 61 g
    "pizza": 107,        # FDC: 1 rebanada (⅛ de pizza 12") ≈ 107 g
    "hot dog": 57,       # FDC: 1 frankfurter ≈ 57 g
    "donut": 60,         # FDC: 1 dona ≈ 60 g
    "cake": 95,          # FDC: 1 rebanada (1/12 de pastel) ≈ 95 g
    "hamburger": 78,     # FDC 170693: 1 sándwich (hamburger single patty) = 78 g
    # === Fast food (FASE 15, verificados) ===
    "french_fries": 117,      # FDC 170698: serving medium = 117 g (small 71, large 154)
    "fried_chicken": 203,     # FDC 170756: 1 breast con piel = 203 g (sin piel 142)
    "chicken_nuggets": 87,    # FDC 172111: 1 serving = 87 g (pieza 18 g)
    "taco": 69,               # FDC 170689: 1 taco = 69 g
    "burrito": 185,           # FDC 172039: 1 burrito = 185 g
    "quesadilla": 157,        # FDC 785882 (FNDDS): 1 quesadilla = 157 g
    "nachos": 80,             # FDC 170291: 1 serving = 80 g
    # === Breakfast (verificados) ===
    "eggs": 61,               # FDC 172187: 1 huevo grande cocido = 61 g
    "bacon": 12,              # FDC 168322: 1 slice pan-fried = 12 g
    "pancakes": 77,           # FDC 175009: 1 pancake 6" = 77 g
    "toast": 22,              # FDC 174925: 1 slice tostada = 22 g
    "bagel": 105,             # FDC 174899: 1 bagel mediano = 105 g
    "waffles": 75,            # FDC 175039: 1 waffle 7" = 75 g
    "oatmeal": 234,           # FDC 173905: 1 cup cocido = 234 g
    # === Main dish (verificados) ===
    "rice": 158,              # FDC 169753: 1 cup cocido = 158 g
    "pasta": 124,             # FDC 168928: 1 cup sin compactar = 124 g
    "salad": 35,              # FDC 787801: 1 cup loosely packed = 35 g
    "steak": 85,              # FDC 169506: 3 oz = 85 g
    "salmon": 170,            # FDC 175168: 1 filete 6 oz = 170 g
    "ice_cream": 66,          # FDC 167575: ½ cup = 66 g
}

# Umbrales de área relativa máscara/imagen (heurística visual, documentada).
SMALL_MAX_RATIO = 0.12
LARGE_MIN_RATIO = 0.30

# Factores de gramos por tamaño (±20 % sobre la referencia).
SIZE_FACTORS = {"small": 0.8, "medium": 1.0, "large": 1.2}
SIZE_RANGES = {
    "small": (0.6, 0.9),
    "medium": (0.8, 1.2),
    "large": (1.1, 1.5),
}

CONFIDENCE = 0.55


def _relative_area(
    detections: list[Detection],
    segmentations: list[SegmentationResult | None],
    index: int,
    image_area: int,
) -> float | None:
    """Área relativa máscara/imagen; fallback al bbox si no hay máscara."""
    if image_area <= 0:
        return None
    segmentation = segmentations[index] if index < len(segmentations) else None
    if segmentation is not None and segmentation.area_pixels > 0:
        return segmentation.area_pixels / image_area
    box = detections[index].bounding_box
    if box.width > 0 and box.height > 0:
        return (box.width * box.height) / image_area
    return None


def _size_from_ratio(ratio: float) -> str:
    if ratio <= SMALL_MAX_RATIO:
        return "small"
    if ratio >= LARGE_MIN_RATIO:
        return "large"
    return "medium"


def _round_int(value: float) -> int:
    return max(1, round(value))


class BasicPortionEstimator(IPortionEstimator):
    """Estimación por porción de referencia + tamaño visual relativo."""

    def estimate(
        self,
        image: Image.Image,
        detections: list[Detection],
        segmentations: list[SegmentationResult | None],
    ) -> list[PortionEstimate | None]:
        image_area = image.width * image.height
        estimates: list[PortionEstimate | None] = []

        for index, detection in enumerate(detections):
            # Normalización de alias: CLIP devuelve canonical con guiones
            # (hot_dog); REFERENCE_GRAMS históricamente usa espacios ("hot dog").
            base_grams = REFERENCE_GRAMS.get(detection.name)
            if base_grams is None:
                base_grams = REFERENCE_GRAMS.get(detection.name.replace("_", " "))
            if base_grams is None:
                estimates.append(PortionEstimate(
                    portion_size="unknown",
                    estimated_grams=None,
                    min_grams=None,
                    max_grams=None,
                    confidence=0.0,
                    method="unavailable",
                ))
                continue

            ratio = _relative_area(detections, segmentations, index, image_area)
            if ratio is None:
                estimates.append(PortionEstimate(
                    portion_size="unknown",
                    estimated_grams=None,
                    min_grams=None,
                    max_grams=None,
                    confidence=0.0,
                    method="unavailable",
                ))
                continue

            size = _size_from_ratio(ratio)
            factor = SIZE_FACTORS[size]
            low, high = SIZE_RANGES[size]

            estimates.append(PortionEstimate(
                portion_size=size,
                estimated_grams=_round_int(base_grams * factor),
                min_grams=_round_int(base_grams * low),
                max_grams=_round_int(base_grams * high),
                confidence=CONFIDENCE,
                method="basic_reference",
            ))

        return estimates