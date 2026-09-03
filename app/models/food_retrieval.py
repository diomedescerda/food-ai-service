"""F42: Food Retrieval Engine — catálogo masivo sin clasificador entrenado.

retrieve(image_or_crop, top_k) -> candidates [{food_id, name, score}]
Índice: embeddings CLIP de texto del catálogo (1.451 entradas FNDDS, CC0),
búsqueda exacta numpy (FAISS documentado para 100k+). Se carga UNA vez.
"""
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("foodai.retrieval")

BASE = Path(__file__).resolve().parents[2]
CATALOG = BASE / "catalog/foods.json"
EMB_DIR = BASE / "catalog/embeddings"
TEMPLATES = ("a photo of {food}", "a picture of {food}", "a close-up photo of {food}")


class FoodRetrieval:
    def __init__(self, enabled: bool = False, clf=None, top_k_default: int = 50):
        self.enabled = enabled
        self.clf = clf
        self.top_k_default = top_k_default
        self.index: np.ndarray | None = None
        self.names: list[str] = []
        self.food_ids: list[str] = []
        self.catalog_size = 0
        if enabled:
            self.load()

    def load(self) -> None:
        try:
            catalog = json.loads(CATALOG.read_text(encoding="utf-8-sig"))
            foods = catalog["foods"]
            self.names = [f["canonical_name"] for f in foods]
            self.food_ids = [f["food_id"] for f in foods]
            self.catalog_size = len(foods)
            mats = [np.load(EMB_DIR / f"clip_text_t{i}.npy") for i in range(len(TEMPLATES))]
            self.index = np.mean(mats, axis=0)
            self.index = self.index / np.linalg.norm(self.index, axis=1, keepdims=True)
            logger.info("retrieval cargado: %d alimentos, índice %s", self.catalog_size, self.index.shape)
        except Exception as exc:  # noqa: BLE001
            logger.error("retrieval NO cargado (fallback): %s", exc)
            self.index = None

    def available(self) -> bool:
        return self.enabled and self.index is not None and self.clf is not None

    def retrieve(self, image_or_crop, top_k: int | None = None) -> list[dict]:
        """Embedding visual CLIP -> índice -> Top-K candidatos ordenados."""
        import torch  # noqa: PLC0415

        if not self.available():
            return []
        k = top_k or self.top_k_default
        inputs = self.clf._processor(images=[image_or_crop], return_tensors="pt")
        with torch.no_grad():
            feats = self.clf._features(self.clf._model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        scores = self.index @ feats.numpy()[0]
        order = np.argsort(-scores)[:k]
        return [{"food_id": self.food_ids[i], "name": self.names[i], "score": float(scores[i])}
                for i in order]