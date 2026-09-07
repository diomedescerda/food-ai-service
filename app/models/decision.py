"""F53: DecisionPolicy — cuándo confiar en el pipeline 5.761 vs legacy.

Tres confianzas SEPARADAS (nunca combinadas):
- visual_confidence: score del retrieval del canonical final
- nutrition_confidence: 0.95 USDA / 0.85 OFF (precomputada)
- portion_confidence: del estimador de porción (externa)

Estados de decisión:
NEW_RESULT_READY / NEW_RESULT_NUTRITION_UNAVAILABLE /
NEW_RESULT_LOW_CONFIDENCE / LEGACY_FALLBACK

Reglas (conservadoras):
- error del pipeline -> LEGACY_FALLBACK (pipeline_error)
- confianza visual < threshold (solo si confidence_enabled) -> LOW_CONFIDENCE
- identificación y nutrición INDEPENDIENTES: nutrition unavailable NO
  revierte la identificación (un alimento puede estar bien identificado sin
  nutrientes descargados todavía).
- Nunca inventa nutrientes ni gramos.
"""
from __future__ import annotations


class DecisionPolicy:
    def __init__(self, min_visual_confidence: float = 0.20, confidence_enabled: bool = False):
        self.min_visual_confidence = min_visual_confidence
        self.confidence_enabled = confidence_enabled

    def decide(
        self,
        visual_confidence: float,
        nutrition_status: str,
        pipeline_error: bool = False,
        nutrition_error: bool = False,
    ) -> dict:
        """Decisión interna. Devuelve estados + fallback_reason."""
        if pipeline_error:
            return {
                "decision": "LEGACY_FALLBACK",
                "fallback_reason": "pipeline_error",
                "identification_ready": False,
                "nutrition_ready": False,
            }
        identification_ready = True
        nutrition_ready = nutrition_status == "NUTRITION_READY" and not nutrition_error

        if self.confidence_enabled and visual_confidence < self.min_visual_confidence:
            return {
                "decision": "NEW_RESULT_LOW_CONFIDENCE",
                "fallback_reason": "low_visual_confidence",
                "identification_ready": False,
                "nutrition_ready": nutrition_ready,
            }
        if nutrition_error:
            return {
                "decision": "NEW_RESULT_NUTRITION_UNAVAILABLE",
                "fallback_reason": "nutrition_error",
                "identification_ready": True,
                "nutrition_ready": False,
            }
        if not nutrition_ready:
            return {
                "decision": "NEW_RESULT_NUTRITION_UNAVAILABLE",
                "fallback_reason": "nutrition_unavailable",
                "identification_ready": True,
                "nutrition_ready": False,
            }
        return {
            "decision": "NEW_RESULT_READY",
            "fallback_reason": None,
            "identification_ready": True,
            "nutrition_ready": True,
        }