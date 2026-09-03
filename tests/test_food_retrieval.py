from app.models.food_retrieval import FoodRetrieval


def test_catalogo_masivo_tiene_mil_entradas():
    import json
    from pathlib import Path
    d = json.loads(Path("catalog/foods.json").read_text(encoding="utf-8-sig"))
    assert len(d["foods"]) >= 1000


def test_indice_cargable():
    import numpy as np
    from pathlib import Path
    emb_dir = Path("catalog/embeddings")
    mats = [np.load(emb_dir / f"clip_text_t{i}.npy") for i in range(3)]
    index = np.mean(mats, axis=0)
    assert index.shape[1] == 512


def test_retrieval_disabled_no_rompe():
    r = FoodRetrieval(enabled=False, clf=None)
    assert r.available() is False
    assert r.retrieve(None) == []
