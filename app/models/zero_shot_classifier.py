"""Clasificador zero-shot con CLIP ViT-B/32 (IFoodClassifier).

Cada detección → crop (bbox + padding configurable) → embedding CLIP →
similaridad con los candidatos del Food Catalog (prompt template
configurable) → nombre canónico + score.

Comportamiento:
- UNKNOWN si ningún candidato supera `clip_threshold` (nunca forzar clase).
- Top-K disponible internamente (candidates) para evaluación/inspección.
- El modelo se carga UNA vez en el startup (lifespan), nunca por request.

Esto es CLIP zero-shot CLASSIFICATION (matching texto-imagen sobre catálogo),
NO embedding/vector search ni RAG: no hay base de datos vectorial ni
recuperación; el catálogo es una lista fija de prompts.
"""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from app.models.classifier_base import ClassificationResult, IFoodClassifier
from app.models.detection import Detection
from app.models.food_catalog import FOOD_CATALOG, all_clip_candidates, candidate_to_canonical


def _iou(a: Detection, b: Detection) -> float:
    """IoU entre dos detecciones (mismas unidades que los bboxes: píxeles)."""
    x1 = max(a.bounding_box.x, b.bounding_box.x)
    y1 = max(a.bounding_box.y, b.bounding_box.y)
    x2 = min(a.bounding_box.x + a.bounding_box.width, b.bounding_box.x + b.bounding_box.width)
    y2 = min(a.bounding_box.y + a.bounding_box.height, b.bounding_box.y + b.bounding_box.height)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a.bounding_box.width * a.bounding_box.height
             + b.bounding_box.width * b.bounding_box.height - inter)
    return inter / union if union > 0 else 0.0


def _contains(a: Detection, b: Detection) -> bool:
    """True si el bbox de `a` contiene al de `b` (>=70% del área de b)."""
    x1 = a.bounding_box.x
    y1 = a.bounding_box.y
    x2 = a.bounding_box.x + a.bounding_box.width
    y2 = a.bounding_box.y + a.bounding_box.height
    bx1 = b.bounding_box.x
    by1 = b.bounding_box.y
    bx2 = b.bounding_box.x + b.bounding_box.width
    by2 = b.bounding_box.y + b.bounding_box.height
    inter_w = max(0, min(x2, bx2) - max(x1, bx1))
    inter_h = max(0, min(y2, by2) - max(y1, by1))
    inter = inter_w * inter_h
    area_b = b.bounding_box.width * b.bounding_box.height
    return area_b > 0 and inter >= 0.7 * area_b


@dataclass(frozen=True)
class ClipTopCandidate:
    name: str
    score: float


class ZeroShotFoodClassifier(IFoodClassifier):
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        threshold: float = 0.20,
        prompt_template: str = "a photo of {food}",
        prompt_templates_extra: tuple[str, ...] = (),
        crop_padding: float = 0.10,
        top_k: int = 5,
        score_by_class: bool = True,
    ):
        self._model_name = model_name
        self._device = device
        self._threshold = threshold
        self._prompt_template = prompt_template
        self._prompt_templates = (prompt_template,) + tuple(prompt_templates_extra)
        self._crop_padding = crop_padding
        self._top_k = top_k
        # FASE 16: scoring por CLASE (máx del mejor candidato) en vez de por
        # candidato global — evitó que hot_dog perdiera contra sandwich en el
        # ranking global (hot_dog: 0/20 → 13/20 en el benchmark).
        self._score_by_class = score_by_class
        self._model = None
        self._processor = None
        self._text_features: torch.Tensor | None = None
        self._prompts: list[str] = []
        self._candidates: list[str] = []

    def load(self) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(self._model_name).to(self._device).eval()
        self._processor = CLIPProcessor.from_pretrained(self._model_name)

        # Precomputar features de texto UNA vez (catálogo fijo). Con más de una
        # plantilla → ensemble por media de scores (FASE 14: mejoró top-1).
        self._candidates = list(all_clip_candidates())
        self._text_features_list = []
        for template in self._prompt_templates:
            prompts = [template.format(food=f) for f in self._candidates]
            text_inputs = self._processor(text=prompts, padding=True, return_tensors="pt").to(self._device)
            with torch.no_grad():
                features = self._features(self._model.get_text_features(**text_inputs))
            self._text_features_list.append(features / features.norm(dim=-1, keepdim=True))

    @staticmethod
    def _features(output: torch.Tensor) -> torch.Tensor:
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        return output.last_hidden_state[:, 0]

    @property
    def model_version(self) -> str:
        return "clip-zero-shot-v1"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and len(self._text_features_list) > 0

    def classify(
        self,
        image: Image.Image,
        detections: list[Detection],
    ) -> list[ClassificationResult | None]:
        """Clasifica cada detección por crop. None si el clasificador no está
        cargado. Deduplica por clase SOLO cuando las regiones se solapan
        (IoU > 0.5, FASE 17): mismas regiones del mismo alimento → una
        instancia; regiones separadas de la misma clase → instancias
        distintas (2 cookies, 2 hamburguesas)."""
        if not self.is_loaded:
            return [None] * len(detections)

        results: list[ClassificationResult | None] = []
        for detection in detections:
            crop = self._crop(image, detection)
            if crop is None:
                results.append(ClassificationResult("unknown", 0.0))
                continue
            top = self._score_crop(crop)
            if not top or top[0].score < self._threshold:
                results.append(ClassificationResult("unknown", 0.0))
            else:
                results.append(ClassificationResult(top[0].name, round(float(top[0].score), 4)))

        # Deduplicación espacial (FASE 17): misma clase + regiones solapadas
        # (IoU > 0.3) o contenidas (caja grande del plato DINO sobre el item)
        # = un solo alimento → conservar la mejor. Instancias separadas
        # (IoU ~0, sin contención) se mantienen: 2 cookies, 2 huevos.
        for i, result in enumerate(results):
            if result is None or result.name == "unknown":
                continue
            for j in range(i):
                other = results[j]
                if other is None or other.name != result.name:
                    continue
                if _iou(detections[i], detections[j]) > 0.3:
                    if result.confidence > other.confidence:
                        results[j] = ClassificationResult("unknown", 0.0)
                    else:
                        results[i] = ClassificationResult("unknown", 0.0)
                    break
                if _contains(detections[i], detections[j]):
                    # La caja i envuelve a j (plato): eliminar la envolvente.
                    results[i] = ClassificationResult("unknown", 0.0)
                    break
                if _contains(detections[j], detections[i]):
                    results[j] = ClassificationResult("unknown", 0.0)
                    break

        return results

    def classify_with_candidates(
        self,
        image: Image.Image,
        detection: Detection,
    ) -> list[ClipTopCandidate]:
        """Top-K candidatos para evaluación/inspección (no afecta producción)."""
        crop = self._crop(image, detection)
        if crop is None:
            return []
        return self._score_crop(crop)

    def _crop(self, image: Image.Image, detection: Detection) -> Image.Image | None:
        """Crop del bbox con padding relativo configurable (0.0 = exacto)."""
        x1 = max(0, int(detection.bounding_box.x - detection.bounding_box.width * self._crop_padding))
        y1 = max(0, int(detection.bounding_box.y - detection.bounding_box.height * self._crop_padding))
        x2 = min(image.width, int(detection.bounding_box.x + detection.bounding_box.width * (1 + self._crop_padding)))
        y2 = min(image.height, int(detection.bounding_box.y + detection.bounding_box.height * (1 + self._crop_padding)))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return image.crop((x1, y1, x2, y2))

    def _score_crop(self, crop: Image.Image) -> list[ClipTopCandidate]:
        inputs = self._processor(images=crop, return_tensors="pt").to(self._device)
        with torch.no_grad():
            image_features = self._features(self._model.get_image_features(**inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            if len(self._text_features_list) == 1:
                scores = (image_features @ self._text_features_list[0].T).squeeze(0).cpu().numpy()
            else:
                # Ensemble: media de scores de todas las plantillas.
                scores = sum(
                    (image_features @ tf.T).squeeze(0).cpu().numpy()
                    for tf in self._text_features_list
                ) / len(self._text_features_list)

        if not self._score_by_class:
            order = np.argsort(-scores)[: self._top_k]
            return [
                ClipTopCandidate(candidate_to_canonical(self._candidates[i]), float(scores[i]))
                for i in order
            ]

        # Scoring por CLASE: score(clase) = máximo de sus candidatos (FASE 16).
        class_scores: dict[str, float] = {}
        index = 0
        for entry in FOOD_CATALOG:
            best = float("-inf")
            for _ in entry.clip_candidates:
                best = max(best, float(scores[index]))
                index += 1
            class_scores[entry.canonical_name] = best
        ranked = sorted(class_scores.items(), key=lambda kv: -kv[1])[: self._top_k]
        return [ClipTopCandidate(name, score) for name, score in ranked]