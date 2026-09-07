# F53 RESULT — Confidence + Fallback

Branch: feature/f53-confidence-fallback · Tests: 142/142 (8 nuevos) · Tree limpio

## Estados de decisión (app/models/decision.py)
NEW_RESULT_READY / NEW_RESULT_NUTRITION_UNAVAILABLE /
NEW_RESULT_LOW_CONFIDENCE / LEGACY_FALLBACK

## Tres confianzas SEPARADAS (nunca combinadas)
- visual_confidence: retrieval_score del canonical final (documentado: 0.298
  en pizza_001 runtime; NO es una calibración — representa el best text
  score del multi-text retrieval)
- nutrition_confidence: 0.95 USDA / 0.85 OFF (precomputada)
- portion_confidence: del estimador de porción (externa, sin inventar)

## Reglas (conservadoras)
- pipeline error -> LEGACY_FALLBACK (pipeline_error)
- visual < threshold -> LOW_CONFIDENCE (SOLO si confidence_enabled)
- identificación y nutrición INDEPENDIENTES: nutrition unavailable NO
  revierte la identificación (alimento identificado sin nutrientes =
  válido)
- Nunca inventa nutrientes ni gramos (portion/calories ausentes del policy)

## Flags
FOOD_AI_CONFIDENCE_ENABLED=false · FOOD_AI_MIN_VISUAL_CONFIDENCE=0.20
(retrieval/nutrition/shadow off — legacy = respuesta)

## Golden cases (tests)
A pizza+READY -> NEW_RESULT_READY ✓ · B id+sin nutrición -> identification
ready + NUTRITION_UNAVAILABLE ✓ · C visual baja -> LOW_CONFIDENCE solo si
enabled ✓ · D DINO/pipeline error -> LEGACY_FALLBACK pipeline_error ✓ ·
E nutrition error -> identificación intacta ✓ · portion ausente -> no
inventa ✓ · confianzas independientes ✓ · determinismo ✓

## Shadow runtime (pizza_001 real)
nutrition_shadow canonical=pizza status=NUTRITION_READY conf=0.95
source=USDA FDC cal=292 protein=14.0 carbs=25.6 fat=14.5 ref=100g
decision_shadow legacy=pizza new=pizza visual_conf=0.298
nutrition_status=NUTRITION_READY nutrition_conf=0.95
decision=NEW_RESULT_READY would_fallback=False reason=None
Respuesta pública: pizza conf=0.9075 (legacy intacta — invariancia ✓)

## Cobertura nutricional actual
1.184/5.761 con nutrientes (20.6%); tandas USDA reanudables (~4 días):
python scripts/f52_nutrition.py --priority (el mecanismo NO depende de la
cobertura completa — identificación y nutrición separadas).

## Producción
Retrieval off, nutrition off, confidence off, specialist off — legacy =
respuesta, 142/142. Recognition intacto.

## Decisión
APPROVE F53 — READY FOR FINAL SHADOW: identificación/nutrición/porción
separadas, fallback con razón, errores experimentales no rompen requests,
shadow registra la decisión completa, sin calibración inventada. Siguiente:
F54 (completar nutrition coverage + shadow final) -> F55 (active rollout).
