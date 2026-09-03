"""Reranking del retrieval con el specialist DINO (pizza/naan) — F44/F45.

Regla final (F45): gate por confianza legacy (< 0.40) + pizza/naan en el
top-K del legacy (K=3) + umbral del specialist (0.75). Si el specialist
confirma con conf >= umbral, el canonical confirmado se promueve al top-1.
Los demás candidatos conservan el score retrieval (sin penalización).
El fallback: excepción del specialist -> retrieval intacto.
"""
from __future__ import annotations

GROUP = {"pizza", "naan"}
SPECIALIST_GATE_CONF = 0.40
SPECIALIST_GATE_TOPK = 3
SPECIALIST_THRESHOLD = 0.75


def specialist_eligible(legacy_conf1: float, legacy_topk: list[str], topk: int = SPECIALIST_GATE_TOPK) -> bool:
    """Gate del F38: solo invocar el specialist si el legacy duda
    (conf1 < 0.40) y pizza/naan están entre sus top-K candidatos."""
    return legacy_conf1 < SPECIALIST_GATE_CONF and any(g in legacy_topk[:topk] for g in GROUP)


def rerank_with_specialist(
    candidates: list[str],
    legacy_conf1: float,
    legacy_topk: list[str],
    spec_class: str,
    spec_conf: float,
    gate_topk: int = SPECIALIST_GATE_TOPK,
    conf_min: float = SPECIALIST_THRESHOLD,
) -> list[str]:
    """Ranking final tras el specialist. Nunca elimina candidatos; solo
    puede promover el canonical confirmado al top-1. El resto sin cambio."""
    present = [c for c in candidates if c in GROUP]
    if not present or not specialist_eligible(legacy_conf1, legacy_topk, gate_topk):
        return list(candidates)
    if spec_conf < conf_min:
        return list(candidates)  # abstención
    if spec_class in candidates:
        final = list(candidates)
        final.remove(spec_class)
        final.insert(0, spec_class)
        return final
    return list(candidates)  # fallback: canonical fuera de candidatos