"""Tests del catálogo escalado (F47): conteo, unicidad de canónicos,
cobertura de embeddings, rebuild del índice, compatibilidad con el 1.451."""
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
CATALOG = BASE / "catalog/foods.json"
EMB_DIR = BASE / "catalog/embeddings"


def _catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8-sig"))


def test_catalog_count() -> None:
    c = _catalog()
    assert c["canonical_foods"] >= 5000  # >= 3x el catálogo 1.451
    assert c["entries"] >= 8500


def test_canonical_uniqueness() -> None:
    c = _catalog()
    names = [f["canonical_name"] for f in c["foods"]]
    assert len(names) == len(set(names))  # sin canónicos duplicados


def test_alias_normalization() -> None:
    c = _catalog()
    for f in c["foods"]:
        for a in f["aliases"]:
            assert a.islower()  # alias normalizado (minúsculas)
    assert sum(len(f["aliases"]) for f in c["foods"]) >= 1000


def test_embedding_coverage() -> None:
    c = _catalog()
    n = c["canonical_foods"]
    for i in range(3):
        arr = np.load(EMB_DIR / f"clip_text_t{i}.npy")
        assert arr.shape == (n, 512)  # cobertura completa + dimensión CLIP


def test_index_rebuild() -> None:
    from app.models.food_retrieval import FoodRetrieval  # noqa: PLC0415

    c = _catalog()
    ret = FoodRetrieval(enabled=True, clf=None)
    assert ret.available() is False  # sin clf -> no disponible (fallback)
    assert ret.catalog_size == len(c["foods"])  # el catálogo se leyó igual
    assert ret.index is not None  # índice reconstruido desde foods.json + npy


def test_legacy_backcompat() -> None:
    c = _catalog()
    names = {f["canonical_name"] for f in c["foods"]}
    aliases = set(a for f in c["foods"] for a in f["aliases"])
    legacy = json.loads((BASE / "catalog/foods_legacy_1451.json").read_text(encoding="utf-8-sig"))
    for f in legacy["foods"]:
        canon = f["canonical_name"]
        # el canónico legacy sigue presente: exacto, como alias, o como
        # canónico más específico que lo contiene (p.ej. "bacon" -> 
        # "bacon for use with vegetables").
        assert (
            canon in names
            or any(canon in a for a in aliases)
            or any(canon in n for n in names)
        ), f"legacy perdido: {canon}"


def test_legacy_smoke_foods_presentes() -> None:
    c = _catalog()
    names = {f["canonical_name"] for f in c["foods"]}
    aliases = set(a for f in c["foods"] for a in f["aliases"])
    for target in ("pizza", "hamburger", "naan", "rice", "fries", "chicken", "pasta", "sandwich"):
        assert target in names or any(target in a for a in aliases), f"falta: {target}"


def test_sources_registradas() -> None:
    c = _catalog()
    sources = {f["source"] for f in c["foods"]}
    assert "USDA FNDDS" in sources
    assert "USDA SR Legacy" in sources
    assert "Open Food Facts" in sources or "USDA FNDDS (legacy 1.451)" in sources