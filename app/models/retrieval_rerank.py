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


def rerank_general(
    candidates: list[dict],
    support_max: float,
    dino_s: tuple[str, float] | None = None,
    w_support: float = 0.5,
    w_alias: float = 0.1,
    w_dino: float = 1.0,
    w_rank: float = 0.0,
    pool_size: int = 200,
) -> list[str]:
    """Reranker general determinista (F48, F49).

    Señales: retrieval_max + support_count + alias_count + specialist DINO
    + rank (posición en el candidate pool — relevante con pools grandes,
    F49: el top-200 diluye el ranking sin el rank). Features ausentes NO
    penalizan. Orden estable para ties.
    """
    out = []
    for c in candidates:
        score = c["max"]
        score += w_support * (c.get("support", 0) / support_max)
        score += w_alias * (0.5 * (c.get("alias", 0) / 20.0))
        if w_rank:
            score += w_rank * (1.0 - (c.get("rank", pool_size) - 1) / pool_size)
        if w_dino and dino_s is not None and c["name"] == dino_s[0]:
            score += w_dino * dino_s[1]
        out.append((c["name"], score))
    out.sort(key=lambda x: -x[1])
    return [n for n, _ in out]


def fuse_views(views: list[list[dict]]) -> list[dict]:
    """Fusión multi-query (F49): union por canonical con max/mean/rank/
    query_count. Cada vista: lista de {name, score, rank}. Dedup por
    canonical — las variantes no consumen posiciones del pool."""
    group: dict[str, dict] = {}
    for view in views:
        for item in view:
            g = group.setdefault(item["name"], {"scores": [], "ranks": []})
            g["scores"].append(float(item["score"]))
            g["ranks"].append(int(item["rank"]))
    out = []
    for name, g in group.items():
        s = g["scores"]
        out.append({
            "name": name, "max": float(max(s)), "mean": float(sum(s) / len(s)),
            "rank": min(g["ranks"]), "queries": len(s),
        })
    return out


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