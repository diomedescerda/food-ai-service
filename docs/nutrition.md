# Nutrición — Food AI

**Estado: FASE 0 — sin base nutricional todavía.**

## Diseño acordado (FASE 5)

Base nutricional propia en la BD compartida de CoppAddresd (schema `foodai.`):

```
Foods             # catálogo (id, name, display_name, category, default_unit)
FoodNutrition     # valores por 100 g o porción (calories, protein, carbs, fat, fiber, sugar, sodium)
FoodAliases       # sinónimos/español ↔ inglés
FoodCategories    # categorías
FoodServingSizes  # porciones de referencia
```

## Principio fundamental

Los valores nutricionales provienen de la BD (deterministas), nunca generados por IA/LLM:

```
arroz cocido: 100 g → 130 kcal · 28 g carbs · 2.7 g protein · 0.3 g fat
180 g → 234 kcal · 50.4 g carbs · 4.86 g protein · 0.54 g fat
```

`NutritionService` recibe `(food, grams)` y devuelve macros escalados. Múltiples referencias por alimento permitidas.

Los resultados se presentan siempre como **estimaciones**, no mediciones exactas.