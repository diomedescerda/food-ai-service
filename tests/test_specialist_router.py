"""F38: tests unitarios del SpecialistRouter (6 casos del prompt)."""
from app.models.specialist_router import SpecialistRouter


def test_confianza_alta_no_invoca_especialista():
    router = SpecialistRouter(enabled=True)
    d = router.decide("hamburger", 0.85, ["hamburger", "pizza", "sandwich"])
    assert not d.use_specialist
    assert d.final_class == "hamburger"


def test_confianza_baja_con_pizza_invoca_y_corrige():
    router = SpecialistRouter(enabled=True)
    d = router.decide("rice", 0.31, ["rice", "pizza", "toast"], specialist_class="pizza", specialist_score=0.81)
    assert d.use_specialist
    assert d.final_class == "pizza"


def test_confianza_baja_sin_pizza_naan_no_invoca():
    router = SpecialistRouter(enabled=True)
    d = router.decide("banana", 0.28, ["banana", "apple", "orange"])
    assert not d.use_specialist
    assert d.final_class == "banana"


def test_especialista_no_disponible_abstiene_a_legacy():
    router = SpecialistRouter(enabled=True)
    d = router.decide("fries", 0.30, ["fries", "pizza", "rice"], specialist_class=None)
    assert d.use_specialist
    assert d.abstained
    assert d.final_class == "fries"


def test_score_bajo_del_especialista_abstiene():
    router = SpecialistRouter(enabled=True)
    d = router.decide("fries", 0.30, ["fries", "pizza", "rice"], specialist_class="naan", specialist_score=0.41)
    assert d.abstained
    assert d.final_class == "fries"


def test_disabled_siempre_legacy():
    router = SpecialistRouter(enabled=False)
    d = router.decide("fries", 0.20, ["fries", "pizza", "rice"], specialist_class="pizza", specialist_score=0.99)
    assert not d.use_specialist
    assert d.final_class == "fries"