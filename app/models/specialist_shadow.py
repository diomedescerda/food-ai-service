"""F39: SpecialistShadow — decisión del especialista SIN modificar la respuesta.

El shadow ejecuta la decisión (router + DINO-base pizza/naan) y registra
telemetría; el resultado oficial SIEMPRE es el legacy. Nunca toca nutrition/
portion/final. Excepciones del specialist -> fallback legacy silencioso.
"""
import logging
import time
from pathlib import Path

from app.models.specialist_router import SpecialistRouter

logger = logging.getLogger("foodai.specialist_shadow")

MODEL_PATH = Path(__file__).resolve().parents[2] / "data/models/f38/pizza_naan_dino_base.pt"
ABSTAIN_SCORE = 0.60


class SpecialistShadow:
    def __init__(self, enabled: bool = False, threshold: float = 0.40,
                 groups: tuple[str, ...] = ("pizza", "naan")):
        self.enabled = enabled
        self.router = SpecialistRouter(enabled=enabled, threshold=threshold, groups=groups)
        self._dino = None
        self._processor = None
        self._head = None
        self._label_map = None
        if enabled and MODEL_PATH.exists():
            try:
                import torch  # noqa: PLC0415
                import torch.nn as nn  # noqa: PLC0415
                from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415

                self._dino = AutoModel.from_pretrained("facebook/dinov2-base").eval()
                self._processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
                ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
                self._head = nn.Linear(768, len(ckpt["classes"]))
                self._head.load_state_dict(ckpt["head"])
                self._head.eval()
                self._label_map = ckpt["label_map"]
                logger.info("specialist shadow cargado: dino-base pizza/naan")
            except Exception as exc:  # noqa: BLE001
                logger.error("specialist shadow NO cargado (fallback legacy): %s", exc)
                self._dino = None

    def available(self) -> bool:
        return self.enabled and self._dino is not None

    def _specialist_predict(self, image):
        import torch  # noqa: PLC0415

        inputs = self._processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            out = self._dino(**inputs)
            feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            probs = torch.softmax(self._head(feats), dim=1)[0]
        inv = {v: k for k, v in self._label_map.items()}
        return inv[int(probs.argmax())], float(probs.max())

    def shadow_evaluate(self, legacy_class: str, legacy_score: float, top3: list[str],
                        image, analysis_id: str, food_index: int) -> dict:
        """Ejecuta la decisión en shadow. Devuelve telemetría. NUNCA modifica el resultado."""
        if not self.available():
            return {"analysis_id": analysis_id, "food_index": food_index,
                    "specialist_available": False, "invoked": False}
        decision = self.router.decide(legacy_class, legacy_score, top3)
        if not decision.use_specialist:
            return {"analysis_id": analysis_id, "food_index": food_index, "invoked": False}
        t0 = time.perf_counter()
        try:
            spec_class, spec_score = self._specialist_predict(image)
        except Exception as exc:  # noqa: BLE001
            logger.error("specialist error (fallback legacy): %s", exc)
            return {"analysis_id": analysis_id, "food_index": food_index, "invoked": True,
                    "error": str(exc), "final_class": legacy_class}
        latency_ms = (time.perf_counter() - t0) * 1000
        decision2 = self.router.decide(legacy_class, legacy_score, top3,
                                       specialist_class=spec_class, specialist_score=spec_score)
        telemetry = {
            "analysis_id": analysis_id, "food_index": food_index, "invoked": True,
            "legacy_prediction": legacy_class, "legacy_score": round(legacy_score, 4),
            "candidate_classes": top3, "specialist_group": "pizza_naan",
            "specialist_prediction": spec_class, "specialist_score": round(spec_score, 4),
            "specialist_abstention": decision2.abstained,
            "would_change_prediction": not decision2.abstained and spec_class != legacy_class,
            "final_class": legacy_class,  # shadow: SIEMPRE legacy
            "latency_ms": round(latency_ms, 1),
        }
        logger.info("specialist_shadow analysis_id=%s food=%s legacy=%s spec=%s would_change=%s abstain=%s lat=%.0fms",
                    analysis_id, food_index, legacy_class, spec_class,
                    telemetry["would_change_prediction"], decision2.abstained, latency_ms)
        return telemetry