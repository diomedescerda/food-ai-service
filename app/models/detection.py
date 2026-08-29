"""Contratos de detección de alimentos."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingBox:
    """Caja delimitadora en PÍXELES (coordenadas de la imagen original),
    formato x/y/width/height — directo para dibujar en el frontend."""

    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Detection:
    name: str
    confidence: float
    bounding_box: BoundingBox


# Clases de comida del dataset COCO (índices de names de YOLO).
# 45 bowl (vajilla) se excluye a propósito: no es un alimento.
FOOD_CLASS_IDS = {46, 47, 48, 49, 50, 51, 52, 53, 54, 55}
FOOD_CLASS_NAMES = {
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
}