"""F52: NutritionService — lookup local de nutrición para los 5.761 canónicos.

Precomputado (nutrition/mappings.json + nutrition/raw/): NUNCA consulta USDA
en runtime (latencia de red cero en inferencia). Estados:
NUTRITION_READY / NUTRITION_UNAVAILABLE. Confianza separada de la visual.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger("foodai.nutrition")

BASE = Path(__file__).resolve().parents[2]
MAPPINGS = BASE / "nutrition/mappings.json"
RAW = BASE / "nutrition/raw"


class NutritionService:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.index: dict[str, dict] = {}
        if enabled:
            self._load()

    def _load(self) -> None:
        try:
            if MAPPINGS.exists():
                self.index = json.loads(MAPPINGS.read_text(encoding="utf-8"))
            logger.info("nutrition service: %d mappings cargados", len(self.index))
        except Exception as exc:  # noqa: BLE001
            logger.error("nutrition service NO cargado (fallback): %s", exc)
            self.index = {}

    def available(self) -> bool:
        return self.enabled and bool(self.index)

    def nutrition_for(self, food_id: str) -> dict:
        """Lookup local. Devuelve el mapping o UNAVAILABLE. Sin red."""
        if not food_id:
            return {"food_id": food_id, "status": "NUTRITION_UNAVAILABLE",
                    "nutrition_confidence": 0.0}
        m = self.index.get(food_id)
        if not m or "nutrients_per_100g" not in m:
            return {"food_id": food_id, "status": "NUTRITION_UNAVAILABLE",
                    "nutrition_confidence": 0.0}
        conf = 0.95 if m.get("source") == "USDA FDC" else 0.85
        return {
            "food_id": food_id,
            "canonical_name": m.get("canonical_name"),
            "status": "NUTRITION_READY",
            "nutrition_confidence": conf,
            "source": m.get("source"),
            "fdc_id": m.get("fdc_id") or m.get("code"),
            "reference_grams": m.get("reference_grams", 100),
            "nutrients_per_100g": m["nutrients_per_100g"],
        }

    def canonical_nutrition(self, canonical: str) -> dict:
        """Resuelve por canonical_name (búsqueda local, sin red)."""
        for food_id, m in self.index.items():
            if m.get("canonical_name") == canonical and "nutrients_per_100g" in m:
                return self.nutrition_for(food_id)
        return {"canonical_name": canonical, "status": "NUTRITION_UNAVAILABLE",
                "nutrition_confidence": 0.0}