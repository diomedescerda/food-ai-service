"""Catálogo de alimentos del clasificador zero-shot (CLIP).

Cada entrada:
- canonical_name: identidad canónica (nombre estable, versionado)
- aliases: variantes de texto para el prompt de CLIP Y para matching
  nutricional futuro (NO todas son el mismo alimento nutricional: el mapping
  nutricional vive en la Nutrition DB y se documenta por entrada)
- category: taxonomía (metadata, no clase YOLO)
- nutrition_key: alias que resolverá INutritionProvider en la DB (None si no
  existe entrada nutricional → nutritionStatus=unavailable)

La identidad visual y la disponibilidad nutricional son problemas separados:
una clase sin nutrition_key se identifica correctamente pero reporta
unavailable (nunca se inventan datos).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FoodCatalogEntry:
    canonical_name: str
    clip_candidates: tuple[str, ...]
    category: str
    nutrition_key: str | None
    aliases: tuple[str, ...] = field(default=())


FOOD_CATALOG: tuple[FoodCatalogEntry, ...] = (
    # === Fast food / restaurante ===
    FoodCatalogEntry("pizza", ("pizza",), "restaurant", "pizza"),
    FoodCatalogEntry("hamburger", ("hamburger", "cheeseburger", "hamburger with a round sesame bun"), "fast_food", None),
    FoodCatalogEntry("french_fries", ("french fries", "fries", "thin fried potato strips", "long golden potato sticks"), "fast_food", None),
    FoodCatalogEntry("hot_dog", ("hot dog", "hot dog sausage in a long bun"), "fast_food", "hot dog"),
    FoodCatalogEntry("sandwich", ("sandwich", "sandwich between two slices of bread"), "restaurant", "sandwich"),
    FoodCatalogEntry("fried_chicken", ("fried chicken", "large fried chicken piece", "battered chicken piece"), "fast_food", None),
    FoodCatalogEntry("chicken_nuggets", ("chicken nuggets",), "fast_food", None),
    FoodCatalogEntry("taco", ("taco", "crispy corn taco shell with meat and lettuce"), "restaurant", None),
    FoodCatalogEntry("burrito", ("burrito",), "restaurant", None),
    FoodCatalogEntry("quesadilla", ("quesadilla", "folded grilled tortilla with melted cheese inside", "quesadilla cut in triangles"), "restaurant", None),
    FoodCatalogEntry("nachos", ("nachos", "nachos with melted cheese on tortilla chips"), "restaurant", None),
    FoodCatalogEntry("steak", ("steak", "grilled steak", "grilled beef steak on a plate"), "restaurant", None),
    FoodCatalogEntry("grilled_chicken", ("grilled chicken", "grilled chicken breast pieces"), "restaurant", None),
    FoodCatalogEntry("salmon", ("salmon", "grilled salmon", "grilled salmon fillet with pink flesh"), "restaurant", None),
    FoodCatalogEntry("rice", ("rice", "cooked rice"), "restaurant", None),
    FoodCatalogEntry("pasta", ("pasta", "spaghetti"), "restaurant", None),
    FoodCatalogEntry("lasagna", ("lasagna",), "restaurant", None),
    FoodCatalogEntry("mac_and_cheese", ("mac and cheese",), "restaurant", None),
    FoodCatalogEntry("salad", ("salad", "green salad"), "restaurant", None),
    FoodCatalogEntry("soup", ("soup", "bowl of soup"), "restaurant", None),
    # === Breakfast ===
    FoodCatalogEntry("eggs", ("eggs", "fried eggs"), "breakfast", None),
    FoodCatalogEntry("bacon", ("bacon", "bacon strips"), "breakfast", None),
    FoodCatalogEntry("toast", ("toast", "slice of toast"), "breakfast", None),
    FoodCatalogEntry("bagel", ("bagel",), "breakfast", None),
    FoodCatalogEntry("pancakes", ("pancakes", "pancake stack with butter and syrup", "flat round pancakes stacked"), "breakfast", None),
    FoodCatalogEntry("waffles", ("waffles", "waffle with square grid pattern", "waffles with syrup in the squares"), "breakfast", None),
    FoodCatalogEntry("cereal", ("cereal", "bowl of cereal"), "breakfast", None),
    FoodCatalogEntry("oatmeal", ("oatmeal", "bowl of oatmeal"), "breakfast", None),
    # === Desserts / snacks ===
    FoodCatalogEntry("donut", ("donut", "doughnut"), "dessert", "donut"),
    FoodCatalogEntry("cake", ("cake", "slice of cake"), "dessert", "cake"),
    FoodCatalogEntry("cookie", ("cookies", "chocolate chip cookies"), "dessert", None),
    FoodCatalogEntry("brownie", ("brownie",), "dessert", None),
    FoodCatalogEntry("ice_cream", ("ice cream", "bowl of ice cream"), "dessert", None),
    # === Frutas / verduras ===
    FoodCatalogEntry("apple", ("apple", "red apple"), "fruit", "apple"),
    FoodCatalogEntry("banana", ("banana",), "fruit", "banana"),
    FoodCatalogEntry("orange", ("orange", "whole orange"), "fruit", "orange"),
    FoodCatalogEntry("broccoli", ("broccoli",), "vegetable", "broccoli"),
    FoodCatalogEntry("carrot", ("carrot",), "vegetable", "carrot"),
)

CATALOG_BY_NAME = {entry.canonical_name: entry for entry in FOOD_CATALOG}


def all_clip_candidates() -> tuple[str, ...]:
    """Todos los candidatos de texto de CLIP (para precomputar features)."""
    return tuple(c for entry in FOOD_CATALOG for c in entry.clip_candidates)


def candidate_to_canonical(candidate: str) -> str:
    """Mapea un candidato CLIP a su nombre canónico."""
    for entry in FOOD_CATALOG:
        if candidate in entry.clip_candidates:
            return entry.canonical_name
    return candidate