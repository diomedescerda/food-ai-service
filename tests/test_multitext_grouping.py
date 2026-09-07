"""Tests del multi-text retrieval (F50): grouping por canonical, best text,
aggregación, dedup, determinismo, aliases ausentes y compatibilidad."""
from app.models.retrieval_rerank import group_text_matches


def test_grouping_max_y_best_text() -> None:
    matches = [
        ("hamburger", 0.24, "hamburger", 30),
        ("hamburger", 0.88, "double hamburger on wheat bun", 17),
        ("fries", 0.30, "french fries", 1),
    ]
    out = group_text_matches(matches, "max")
    assert out[0]["name"] == "hamburger"
    assert out[0]["score"] == 0.88
    assert out[0]["best_text"] == "double hamburger on wheat bun"
    assert out[0]["best_rank"] == 17
    assert out[0]["queries"] == 2


def test_aggregation_top2mean() -> None:
    matches = [
        ("pizza", 0.20, "pizza", 40),
        ("pizza", 0.30, "pizza margherita", 5),
        ("pizza", 0.26, "pizza pepperoni", 10),
    ]
    out = group_text_matches(matches, "top2mean")
    assert abs(out[0]["score"] - 0.28) < 1e-9  # mean(top2: 0.30, 0.26)


def test_dedup_canonical() -> None:
    matches = [("pizza", s, f"pizza variant {i}", i) for i, s in enumerate([0.1, 0.2, 0.3], 1)]
    out = group_text_matches(matches)
    assert len(out) == 1  # un solo canonical pese a 3 textos
    assert out[0]["queries"] == 3


def test_missing_alias_no_rompe() -> None:
    assert group_text_matches([]) == []
    out = group_text_matches([("rice", 0.25, "rice", 2)])
    assert out[0]["name"] == "rice" and out[0]["queries"] == 1


def test_deterministic_order() -> None:
    matches = [("a", 0.2, "a", 1), ("b", 0.19, "b", 2), ("c", 0.18, "c", 3)]
    assert group_text_matches(matches) == group_text_matches(matches)


def test_tie_estable() -> None:
    matches = [("a", 0.2, "a", 1), ("b", 0.2, "b", 2)]
    out = group_text_matches(matches)
    assert out[0]["name"] == "a"  # orden de entrada en ties


def test_multiples_templates_se_funden() -> None:
    # 3 templates del mismo texto = 3 matches del mismo canonical
    matches = [("hamburger", s, f"hamburger t{t}", r) for t, (s, r) in enumerate([(0.3, 1), (0.29, 2), (0.28, 3)])]
    out = group_text_matches(matches)
    assert len(out) == 1 and out[0]["queries"] == 3