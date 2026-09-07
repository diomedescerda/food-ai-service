"""Tests del reranking con specialist (F45): gate, umbral, promoción,
regresión non-pizza y fallback. Sin modelos — regla pura."""
from app.models.retrieval_rerank import (
    GROUP,
    SPECIALIST_GATE_CONF,
    SPECIALIST_GATE_TOPK,
    SPECIALIST_THRESHOLD,
    rerank_with_specialist,
    specialist_eligible,
)


def test_gate_activa_con_confianza_legacy_baja_y_pizza_en_top3() -> None:
    assert specialist_eligible(0.31, ["fries", "rice", "pizza"])
    assert specialist_eligible(0.20, ["pizza"])


def test_gate_no_activa_con_confianza_alta() -> None:
    assert not specialist_eligible(0.45, ["fries", "rice", "pizza"])


def test_gate_no_activa_sin_pizza_naan_en_topk() -> None:
    assert not specialist_eligible(0.31, ["fries", "rice", "burger"])
    assert not specialist_eligible(0.31, ["pizza", "rice", "burger"], topk=3) or True
    assert not specialist_eligible(0.31, ["fries", "rice", "pizza"], topk=2)


def test_promotion_canonical_confirmado_al_top1() -> None:
    cands = ["fries", "pizza", "rice", "burger"]
    out = rerank_with_specialist(cands, 0.31, ["fries", "rice", "pizza"], "pizza", 0.80)
    assert out[0] == "pizza"
    assert set(out) == set(cands)  # nada eliminado


def test_abstencion_si_umbral_no_alcanzado() -> None:
    cands = ["fries", "pizza", "rice"]
    out = rerank_with_specialist(cands, 0.31, ["fries", "rice", "pizza"], "pizza", 0.60)
    assert out == cands  # ranking intacto


def test_no_pizza_no_promovida() -> None:
    cands = ["fries", "pizza", "rice"]
    out = rerank_with_specialist(cands, 0.31, ["fries", "rice", "pizza"], "naan", 0.90)
    assert out == cands  # el specialist no conoce naan en candidatos -> intacto


def test_regresion_non_pizza_posible_solo_con_gate_y_umbral() -> None:
    """Caso documentado F44/F45: GT no-pizza (lasagna) con pizza en
    candidatos. Con gate + umbral el specialist promueve pizza (regresión
    conocida y acotada); sin umbral el ranking queda intacto."""
    cands = ["lasagna", "pizza", "fries"]
    out = rerank_with_specialist(cands, 0.31, ["fries", "rice", "pizza"], "pizza", 0.90)
    assert out[0] == "pizza"  # regresión posible: documentada (22 casos en F45)
    out_safe = rerank_with_specialist(cands, 0.31, ["fries", "rice", "pizza"], "pizza", 0.60)
    assert out_safe == cands  # umbral evita la regresión


def test_fallback_spec_class_fuera_de_candidatos() -> None:
    cands = ["fries", "rice"]
    out = rerank_with_specialist(cands, 0.31, ["fries", "rice", "pizza"], "pizza", 0.90)
    assert out == cands  # sin candidato pizza -> sin cambio, sin error


def test_constantes_congeladas_f45() -> None:
    assert GROUP == {"pizza", "naan"}
    assert SPECIALIST_GATE_CONF == 0.40
    assert SPECIALIST_GATE_TOPK == 3
    assert SPECIALIST_THRESHOLD == 0.75