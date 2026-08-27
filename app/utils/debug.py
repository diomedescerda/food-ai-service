"""Imágenes de debug para desarrollo: overlay de bounding boxes + labels."""

from pathlib import Path

from PIL import Image, ImageDraw

from app.models.detection import Detection

_BOX_COLOR = (232, 89, 12)
_LABEL_BG = (232, 89, 12)
_LABEL_FG = (255, 255, 255)


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Copia la imagen con bounding boxes + labels + confidence dibujados."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        box = det.bounding_box
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
    output_dir: str, analysis_id: str, image: Image.Image, detections: list[Detection]
) -> str | None:
    """Guarda la imagen anotada en {output_dir}/{analysis_id}.jpg. None si off."""
    if not output_dir:
        return None
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{analysis_id}.jpg"
    draw_detections(image, detections).save(path, "JPEG", quality=90)
    return str(path)