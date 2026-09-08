"""Tests del recall expansion (F49): fusión multi-query, dedup por canonical,
query_count, determinismo y compatibilidad con el reranker F48."""
from app.models.retrieval_rerank import fuse_views, rerank_general


def _view(pairs):
    return [{"name": n, "score": s, "rank": r} for n, s, r in pairs]


def test_fusion_max_mean_query_count() -> None:
    v1 = _view([("hamburger", 0.30, 1), ("fries", 0.25, 2)])
    v2 = _view([("hamburger", 0.28, 3), ("rice", 0.26, 1)])
    out = fuse_views([v1, v2])
    by_name = {c["name"]: c for c in out}
    assert by_name["hamburger"]["max"] == 0.30
    assert abs(by_name["hamburger"]["mean"] - 0.29) < 1e-9
    assert by_name["hamburger"]["queries"] == 2
    assert by_name["hamburger"]["rank"] == 1  # best_rank
    assert by_name["fries"]["queries"] == 1


def test_dedup_canonical_sin_variantes() -> None:
    v1 = _view([("pizza", 0.30, 1), ("pizza margherita", 0.29, 2), ("pizza pepperoni", 0.28, 3)])
    out = fuse_views([v1])
    assert len(out) == 3  # entries distintos (el grouping los junta aparte)
    names = {c["name"] for c in out}
    assert "pizza" in names and "pizza margherita" in names


def test_fusion_determinista() -> None:
    v1 = _view([("a", 0.2, 1), ("b", 0.19, 2)])
    v2 = _view([("b", 0.18, 1), ("c", 0.17, 2)])
    a = fuse_views([v1, v2])
    b = fuse_views([v1, v2])
    assert a == b


def test_fusion_vacio() -> None:
    assert fuse_views([]) == []
    assert fuse_views([[], []]) == []


def test_reranker_k_configurable_pool() -> None:
    cands = [
        {"name": "x", "max": 0.30, "rank": 1, "support": 1, "alias": 0},
        {"name": "gen", "max": 0.28, "rank": 150, "support": 23, "alias": 20},
    ]
    # pool grande: el rank (w_rank alto) modera a los genéricos del fondo
    out = rerank_general(cands, support_max=50, w_rank=1.0, pool_size=200)
    assert out[0] == "x"
    # sin el rank, el genérico del fondo gana (el comportamiento F48)
    out2 = rerank_general(cands, support_max=50)
    assert out2[0] == "gen"


def test_backcompat_f48_sin_rank() -> None:
    # la llamada sin w_rank == el comportamiento F48 exacto
    cands = [
        {"name": "hamburger", "max": 0.24, "rank": 15, "support": 23, "alias": 23},
        {"name": "ladoo", "max": 0.275, "rank": 1, "support": 1, "alias": 0},
    ]
    out = rerank_general(cands, support_max=50)
    assert out[0] == "hamburger"  # el genérico del food-us sigue ganando