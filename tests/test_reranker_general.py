"""Tests del reranker general (F48): features ausentes, ties, determinismo,
specialist unavailable, fallback de orden, promoción del canónico genérico."""
from app.models.retrieval_rerank import rerank_general


def _cands(names_max_support_alias):
    return [
        {"name": n, "max": m, "support": s, "alias": a}
        for n, m, s, a in names_max_support_alias
    ]


def test_missing_features_no_penalizan() -> None:
    cands = _cands([("pizza", 0.24, 20, 23), ("pizza margherita", 0.26, 1, 0)])
    out = rerank_general(cands, support_max=50)
    # la pizza genérica (sin specialist, sin penalizar por features ausentes)
    assert out[0] == "pizza"


def test_tie_estable() -> None:
    cands = _cands([("a", 0.20, 1, 0), ("b", 0.20, 1, 0), ("c", 0.20, 1, 0)])
    out = rerank_general(cands, support_max=50)
    assert out == ["a", "b", "c"]  # orden de entrada preservado


def test_deterministic_output() -> None:
    cands = _cands([("x", 0.30, 5, 2), ("y", 0.28, 3, 1), ("z", 0.26, 10, 4)])
    a = rerank_general(cands, support_max=50)
    b = rerank_general(cands, support_max=50)
    assert a == b


def test_specialist_unavailable_no_penaliza() -> None:
    cands = _cands([("pizza", 0.24, 20, 23), ("rice", 0.25, 1, 0)])
    # sin dino_s: el ranking no cambia vs el mismo con dino None
    out = rerank_general(cands, support_max=50, dino_s=None, w_dino=1.0)
    assert out[0] == "pizza"


def test_specialist_promueve_cuando_aplica() -> None:
    cands = _cands([("naan", 0.20, 1, 1), ("pizza", 0.24, 1, 1)])
    out = rerank_general(cands, support_max=50, dino_s=("pizza", 0.87))
    assert out[0] == "pizza"  # el DINO confirma pizza -> top-1


def test_specialist_no_aplica_a_otros_canonicales() -> None:
    cands = _cands([("hamburger", 0.25, 20, 23), ("fries", 0.24, 1, 0)])
    out = rerank_general(cands, support_max=50, dino_s=("pizza", 0.9))
    assert out[0] == "hamburger"  # el score DINO solo suma al canonical que confirma


def test_support_estructural_promueve_generico() -> None:
    cands = _cands([("hamburger double bacon", 0.26, 1, 0), ("hamburger", 0.24, 23, 23)])
    out = rerank_general(cands, support_max=50)
    assert out[0] == "hamburger"