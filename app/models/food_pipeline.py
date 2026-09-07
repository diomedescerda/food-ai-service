"""F51: pipeline único de reconocimiento de 5.761 alimentos (integrado).

Flujo exacto: imagen/crop -> CLIP embedding -> multi-text retrieval ->
canonical grouping -> general reranker -> specialist DINO pizza/naan ->
canonical final. Todos los componentes son los validados en F43-F50:
- índice multi-text (canonical + aliases, 3 templates) — F50
- group_text_matches — F50
- rerank_general — F48 (top-50, sin w_rank: comportamiento probado)
- specialist DINO pizza/naan — F45 (threshold 0.75, gate top-3, conf 0.40)

Seguridad: cualquier excepción interna -> analyze_food devuelve un resultado
marcado como fallback (el caller decide usar legacy). Nunca lanza al
consumidor.
"""
import json
import logging
import time
from pathlib import Path

import numpy as np

from app.models.retrieval_rerank import group_text_matches, rerank_general

logger = logging.getLogger("foodai.pipeline")

BASE = Path(__file__).resolve().parents[2]
EMB_DIR = BASE / "catalog/embeddings/multitext"
TEMPLATES = ("a photo of {food}", "a picture of {food}", "a close-up photo of {food}")
SPECIALIST_MODEL = BASE / "data/models/f38/pizza_naan_dino_base.pt"
SPECIALIST_GROUPS = ("pizza", "naan")
SPECIALIST_THRESHOLD = 0.75
SPECIALIST_GATE_TOPK = 3
SPECIALIST_GATE_CONF = 0.40


class FoodPipeline:
    """Pipeline experimental 5.761 alimentos. Interfaz analyze_food(image)."""

    def __init__(self, clf, enabled: bool = True):
        self.enabled = enabled
        self.clf = clf
        self.texts: list[dict] = []
        self.index: np.ndarray | None = None
        self.support: dict[str, int] = {}
        self.alias_count: dict[str, int] = {}
        self.support_max = 1.0
        self._dino = None
        self._processor = None
        self._head = None
        self._label_map = None
        if enabled:
            self._load()

    def _load(self) -> None:
        try:
            catalog = json.loads((BASE / "catalog/foods.json").read_text(encoding="utf-8-sig"))
            for f in catalog["foods"]:
                # F48: el support/alias venían del catálogo 1.451 (máx ~23
                # aliases, support = variantes). El v4 colapsó a 1 entry por
                # canonical y el cross-check legacy subió aliases > 20 —
                # restaurar el rango calibrado del reranker F48.
                n_aliases = min(len(f["aliases"]), 20)
                self.support[f["canonical_name"]] = 1 + n_aliases
                self.alias_count[f["canonical_name"]] = n_aliases
            self.support_max = max(self.support.values()) or 1.0
            self.texts = json.loads((EMB_DIR / "texts.json").read_text(encoding="utf-8"))
            mats = [np.load(EMB_DIR / f"text_emb_t{i}.npy") for i in range(len(TEMPLATES))]
            self.index = np.mean(mats, axis=0)
            self.index = self.index / np.linalg.norm(self.index, axis=1, keepdims=True)
            logger.info("pipeline F51: índice multi-text %s (%d textos)", self.index.shape, len(self.texts))
        except Exception as exc:  # noqa: BLE001
            logger.error("pipeline F51 NO cargado (fallback legacy): %s", exc)
            self.index = None
        try:
            self._load_dino()
        except Exception as exc:  # noqa: BLE001
            logger.error("pipeline F51: DINO no disponible (solo retrieval): %s", exc)
            self._dino = None

    def _load_dino(self) -> None:
        import torch  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415
        from transformers import AutoImageProcessor, AutoModel  # noqa: PLC0415

        self._dino = AutoModel.from_pretrained("facebook/dinov2-base").eval()
        self._processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        ckpt = torch.load(SPECIALIST_MODEL, map_location="cpu", weights_only=False)
        self._head = nn.Linear(768, len(ckpt["classes"]))
        self._head.load_state_dict(ckpt["head"])
        self._head.eval()
        self._label_map = ckpt["label_map"]

    def available(self) -> bool:
        return self.enabled and self.index is not None and self.clf is not None

    def _image_embedding(self, image):
        import torch  # noqa: PLC0415

        inputs = self.clf._processor(images=image, return_tensors="pt")
        with torch.no_grad():
            feats = self.clf._features(self.clf._model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.numpy()[0]

    def _dino_predict(self, image):
        import torch  # noqa: PLC0415

        inputs = self._processor(images=[image], return_tensors="pt")
        with torch.no_grad():
            out = self._dino(**inputs)
            feats = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            probs = torch.softmax(self._head(feats), dim=1)[0]
        inv = {v: k for k, v in self._label_map.items()}
        return inv[int(probs.argmax())], float(probs.max())

    def analyze_food(self, image, top_k: int = 50, legacy_conf1: float = 0.0,
                     legacy_top3: list[str] | None = None) -> dict:
        """Pipeline completo. Devuelve el canonical final + telemetría interna.
        Fallback: cualquier error -> {"fallback": True} (el caller usa legacy)."""
        result = {"fallback": False, "specialist": {"used": False}}
        if not self.available():
            result["fallback"] = True
            return result
        t0 = time.perf_counter()
        try:
            emb = self._image_embedding(image)
            emb_t = (time.perf_counter() - t0) * 1000

            # multi-text retrieval (top_k text matches) + grouping por canonical
            scores = self.index @ emb
            order = np.argsort(-scores)[:top_k]
            matches = [(self.texts[i]["canonical"], float(scores[i]), self.texts[i]["text"], ei + 1)
                       for ei, i in enumerate(order)]
            cands = group_text_matches(matches, "max")
            for c in cands:
                c["support"] = self.support.get(c["name"], 0)
                c["alias"] = self.alias_count.get(c["name"], 0)
            result["retrieval"] = {
                "top1": cands[0]["name"] if cands else None,
                "top5": [c["name"] for c in cands[:5]],
                "top10": [c["name"] for c in cands[:10]],
            }
            result["best_text"] = cands[0]["best_text"] if cands else None

            # general reranker (F48: top-50, support + alias)
            ranking = rerank_general(cands, self.support_max)
            result["reranker"] = {"used": True, "top1": ranking[0] if ranking else None}
            final = ranking

            # specialist DINO pizza/naan (regla F45 congelada)
            spec_used = False
            if self._dino is not None and any(g in final[:SPECIALIST_GATE_TOPK] for g in SPECIALIST_GROUPS):
                top3 = legacy_top3 or []
                if legacy_conf1 < SPECIALIST_GATE_CONF and any(g in top3 for g in SPECIALIST_GROUPS):
                    sc, sf = self._dino_predict(image)
                    if sf >= SPECIALIST_THRESHOLD and sc in final:
                        final = list(final)
                        final.remove(sc)
                        final.insert(0, sc)
                        spec_used = True
                        result["specialist"] = {"used": True, "prediction": sc,
                                                "confidence": round(sf, 4)}
            result["canonical_name"] = final[0] if final else None
            result["confidence"] = {
                "retrieval_score": round(cands[0]["score"], 4) if cands else 0.0,
                "reranker_score": None,
                "specialist_score": result["specialist"].get("confidence"),
            }
            result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            result["embedding_ms"] = round(emb_t, 1)
            result["specialist_used"] = spec_used
        except Exception as exc:  # noqa: BLE001 — el pipeline jamás rompe el request
            logger.error("pipeline F51 error (fallback legacy): %s", exc)
            result["fallback"] = True
        return result