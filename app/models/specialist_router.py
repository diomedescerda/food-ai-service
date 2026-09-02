"""F38: SpecialistRouter — decisión de uso del especialista (abstracción pura).

Reglas (gate B de F37/F38):
- specialist disabled -> legacy siempre
- conf legacy >= threshold -> legacy
- conf baja pero pizza/naan NO en top-3 -> legacy
- conf baja + pizza/naan en top-3 -> invocar DINO; si DINO score >= abstain
  score -> DINO; si no -> legacy (abstención)
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RouterDecision:
    use_specialist: bool
    final_class: str
    abstained: bool = False


class SpecialistRouter:
    def __init__(self, enabled: bool, threshold: float = 0.40, abstain_score: float = 0.60,
                 groups: tuple[str, ...] = ("pizza", "naan")):
        self.enabled = enabled
        self.threshold = threshold
        self.abstain_score = abstain_score
        self.groups = groups

    def decide(self, legacy_class: str, legacy_score: float, top3: list[str],
               specialist_class: str | None = None, specialist_score: float | None = None) -> RouterDecision:
        if not self.enabled:
            return RouterDecision(use_specialist=False, final_class=legacy_class)
        if legacy_score >= self.threshold:
            return RouterDecision(use_specialist=False, final_class=legacy_class)
        if not any(g in top3 for g in self.groups):
            return RouterDecision(use_specialist=False, final_class=legacy_class)
        if specialist_class is None or specialist_score is None:
            return RouterDecision(use_specialist=True, final_class=legacy_class, abstained=True)
        if specialist_score < self.abstain_score:
            return RouterDecision(use_specialist=True, final_class=legacy_class, abstained=True)
        return RouterDecision(use_specialist=True, final_class=specialist_class)