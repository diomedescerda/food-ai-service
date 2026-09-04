"""F46: RetrievalShadow — pipeline retrieval+grouping+rerank en shadow real.

Ejecuta el pipeline completo (retrieval top-50 -> canonical grouping ->
specialist DINO pizza/naan cuando aplica -> rerank) y registra telemetría.
El resultado oficial del API SIEMPRE es el legacy; el shadow jamás lo toca.
Fallback: cualquier excepción -> telemetría de error, request intacto.
"""
import logging
import time
from pathlib import Path

from app.models.retrieval_rerank import rerank_with_specialist, specialist_eligible

logger = logging.getLogger("foodai.retrieval_shadow")

MODEL_PATH = Path(__file__).resolve().parents[2] / "data/models/f38/pizza_naan_dino_base.pt"
CONF_MIN = 0.75
GATE_TOPK = 3
GATE_CONF = 0.40


class RetrievalShadow:
    def __init__(self, enabled: bool = False, clf=None, retrieval=None):
        self.enabled = enabled
        self.clf = clf
        self.retrieval = retrieval
        self._dino = None
        self._processor = None
        self._head = None
        self._label_map = None
        self.catalog_size = 0
        if enabled:
            self._load_dino()

    def _load_dino(self) -> None:
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
            logger.info("retrieval shadow: DINO-base pizza/naan cargado")
        except Exception as exc:  # noqa: BLE001 — fallback silencioso
            logger.error("retrieval shadow: DINO NO cargado (fallback): %s", exc)
            self._dino = None

    def available(self) -> bool:
        return self.enabled and self._dino is not None and self.retrieval is not None and self.retrieval.available()

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

    def shadow_evaluate(self, image, analysis_id: str, legacy_conf1: float = 0.0,
                        legacy_top3: list[str] | None = None) -> dict:
        """Pipeline completo en shadow. Devuelve telemetría; NUNCA muta nada."""
        base = {"analysis_id": analysis_id, "shadow_available": False}
        if not self.available():
            return base
        top3 = legacy_top3 or []
        telemetry = {
            "analysis_id": analysis_id, "shadow_available": True,
            "catalog_size": self.retrieval.catalog_size,
            "legacy_conf1": round(legacy_conf1, 4), "legacy_top3": top3,
            "specialist_called": False, "specialist_abstain": False,
            "specialist_confidence": None, "specialist_latency_ms": 0,
            "fallback": False, "error": None,
        }
        t0 = time.perf_counter()
        try:
            rank = self.retrieval.retrieve(image, top_k=50)
            groups: dict[str, list[float]] = {}
            for e in rank:
                groups.setdefault(e["name"], []).append(e["score"])
            cands = sorted([(n, max(s), len(s)) for n, s in groups.items()], key=lambda x: -x[1])
            telemetry["retrieval_top1"] = cands[0][0]
            telemetry["retrieval_top5"] = [n for n, _, _ in cands[:5]]
            telemetry["retrieval_top10"] = [n for n, _, _ in cands[:10]]
            telemetry["retrieval_top20"] = [n for n, _, _ in cands[:20]]
            telemetry["retrieval_top50"] = [n for n, _, _ in cands[:50]]
            telemetry["canonical_top1"] = cands[0][0]
            telemetry["canonical_count"] = len(cands)
            reranked = [n for n, _, _ in cands]
            if specialist_eligible(legacy_conf1, top3, GATE_TOPK):
                t1 = time.perf_counter()
                spec_class, spec_conf = self._specialist_predict(image)
                telemetry["specialist_latency_ms"] = round((time.perf_counter() - t1) * 1000, 1)
                telemetry["specialist_prediction"] = spec_class
                telemetry["specialist_confidence"] = round(spec_conf, 4)
                telemetry["specialist_called"] = True
                telemetry["specialist_abstain"] = spec_conf < CONF_MIN
                reranked = rerank_with_specialist(
                    [n for n, _, _ in cands], legacy_conf1, top3,
                    spec_class, spec_conf, gate_topk=GATE_TOPK, conf_min=CONF_MIN,
                )
            telemetry["reranked_top1"] = reranked[0] if reranked else None
            telemetry["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        except Exception as exc:  # noqa: BLE001 — el shadow jamás rompe la respuesta
            telemetry["fallback"] = True
            telemetry["error"] = str(exc)[:200]
            telemetry["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            logger.error("retrieval shadow error (ignorado): %s", exc)
        return telemetry