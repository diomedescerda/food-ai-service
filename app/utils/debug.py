"""Imágenes de debug para desarrollo: overlay de máscaras + bounding boxes + labels + depth."""

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.models.detection import Detection
from app.models.segmenter_base import SegmentationResult

_BOX_COLOR = (232, 89, 12)
_LABEL_BG = (232, 89, 12)
_LABEL_FG = (255, 255, 255)
_MASK_COLOR = (255, 180, 60, 110)


def _mask_overlay(image: Image.Image, box, mask_b64: str) -> Image.Image:
    """Pinta la máscara (PNG b64 recortada al bbox) semi-transparente sobre la imagen.

    El overlay es del tamaño de la imagen completa; la máscara se coloca en la
    posición del bbox y se compone (alpha_composite exige tamaños iguales).
    """
    mask_img = Image.open(BytesIO(base64.b64decode(mask_b64))).convert("L")
    mask_alpha = mask_img.point(lambda p: 255 if p > 0 else 0)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_rgba = Image.new("RGBA", (box.width, box.height), _MASK_COLOR)
    mask_rgba.putalpha(mask_alpha)
    overlay.paste(mask_rgba, (box.x, box.y))

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def draw_detections(
    image: Image.Image,
    detections: list[Detection],
    segmentations: list[SegmentationResult | None] | None = None,
) -> Image.Image:
    """Copia la imagen con máscaras, bounding boxes, labels y confidence."""
    annotated = image.convert("RGB").copy()
    segs = segmentations if segmentations is not None else [None] * len(detections)
    draw = ImageDraw.Draw(annotated)
    for det, seg in zip(detections, segs):
        box = det.bounding_box
        if seg is not None:
            annotated = _mask_overlay(annotated, box, seg.mask)
            draw = ImageDraw.Draw(annotated)
        draw.rectangle(
            (box.x, box.y, box.x + box.width, box.y + box.height),
            outline=_BOX_COLOR,
            width=3,
        )
        label = f"{det.name} {det.confidence:.2f}"
        draw.rectangle(
            (box.x, box.y - 18, box.x + draw.textlength(label) + 8, box.y),
            fill=_LABEL_BG,
        )
        draw.text((box.x + 4, box.y - 16), label, fill=_LABEL_FG)
    return annotated


def save_debug_image(
    output_dir: str,
    analysis_id: str,
    image: Image.Image,
    detections: list[Detection],
    segmentations: list[SegmentationResult | None] | None = None,
) -> str | None:
    """Guarda la imagen anotada en {output_dir}/{analysis_id}.jpg. None si off."""
    if not output_dir:
        return None
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{analysis_id}.jpg"
    draw_detections(image, detections, segmentations).save(path, "JPEG", quality=90)
    return str(path)


def save_depth_image(output_dir: str, analysis_id: str, depth) -> str | None:
    """Guarda el depth map (jet) en {output_dir}/{analysis_id}_depth.jpg."""
    if not output_dir or depth is None:
        return None
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{analysis_id}_depth.jpg"

    data = depth.data
    if data.max() > data.min():
        normalized = (data - data.min()) / (data.max() - data.min())
    else:
        normalized = data
    colored = np.zeros((*normalized.shape, 3), dtype=np.uint8)
    cmap = np.array(
        [[0, 0, 131], [0, 60, 170], [5, 113, 176], [89, 161, 79], [156, 187, 26], [251, 232, 0], [252, 141, 89], [219, 64, 53], [150, 0, 24]],
        dtype=np.uint8,
    )
    indices = np.clip((normalized * (len(cmap) - 1)).astype(int), 0, len(cmap) - 1)
    colored[:] = cmap[indices]
    Image.fromarray(colored, "RGB").save(path, "JPEG", quality=90)
    return str(path)