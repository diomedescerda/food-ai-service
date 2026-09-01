"""FASE 25: genera el Food Master Catalog (data/catalogs/food_master.json).

Fuente única consolidada: food_catalog.py (visual 38) + JSON curado USDA
(backennd) + REFERENCE_GRAMS (porción). Las clases NUEVAS de la expansión
viven en CATALOG_EXPANSION (lista curada con candidates descriptivos).

Estados derivados de los datos (nunca inventados):
- PRODUCTION_READY: visual + nutrition + portion
- NUTRITION_READY:    visual + nutrition, sin porción
- VISUAL_ONLY:        visual, sin nutrition
- REVIEW_REQUIRED:    visual con mapping ambiguo/no verificado
- NO_RELIABLE_NUTRITION: sin equivalencia nutricional defendible

Uso: python scripts/build_food_master.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.food_catalog import FOOD_CATALOG  # noqa: E402
from app.models.basic_portion_estimator import REFERENCE_GRAMS  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
BACK = BASE.parent / "coppAddresdBack"
CURATED = BACK / "src/CoppAddresd.Api/Seeders/data/food_usda_curated.json"
OUT = BASE / "data/catalogs/food_master.json"

# ── Expansión FASE 25: clases nuevas (curadas, con candidates descriptivos) ──
# canonical: (category, candidates...)
CATALOG_EXPANSION: dict[str, tuple[str, tuple[str, ...]]] = {
    # Desayunos / panadería
    "english_muffin": ("breakfast", ("english muffin", "toasted english muffin")),
    "croissant": ("breakfast", ("croissant", "butter croissant pastry")),
    "muffin": ("breakfast", ("muffin", "blueberry muffin", "bakery muffin")),
    "cereal_bowl": ("breakfast", ("cereal with milk", "bowl of breakfast cereal")),
    "granola": ("breakfast", ("granola", "granola with yogurt")),
    "yogurt": ("breakfast", ("yogurt", "plain yogurt cup")),
    "cottage_cheese": ("breakfast", ("cottage cheese", "cottage cheese bowl")),
    "french_toast": ("breakfast", ("french toast", "french toast with syrup")),
    "scrambled_eggs": ("breakfast", ("scrambled eggs", "fluffy scrambled eggs")),
    "omelette": ("breakfast", ("omelette", "cheese omelette")),
    "sausage_patty": ("breakfast", ("breakfast sausage patty", "pork sausage links")),
    "hash_browns": ("breakfast", ("hash browns", "crispy hash brown potato")),
    "grits": ("breakfast", ("grits", "southern grits bowl")),
    "crepes": ("breakfast", ("crepes", "french crepes with filling")),
    "churros": ("dessert", ("churros", "cinnamon churros")),
    # Sándwiches / wraps / hamburguesas
    "club_sandwich": ("restaurant", ("club sandwich", "triple decker club sandwich")),
    "grilled_cheese": ("restaurant", ("grilled cheese sandwich", "toasted cheese sandwich")),
    "reuben": ("restaurant", ("reuben sandwich", "corned beef reuben")),
    "sub_sandwich": ("restaurant", ("sub sandwich", "hoagie sub")),
    "wrap": ("restaurant", ("wrap sandwich", "tortilla wrap with filling")),
    "pulled_pork": ("restaurant", ("pulled pork sandwich", "bbq pulled pork")),
    "fish_and_chips": ("restaurant", ("fish and chips", "battered fish with fries")),
    "chicken_wings": ("restaurant", ("chicken wings", "buffalo chicken wings")),
    "chicken_tenders": ("restaurant", ("chicken tenders", "fried chicken strips")),
    "corn_dog": ("restaurant", ("corn dog", "corn dog on a stick")),
    "cheeseburger": ("restaurant", ("cheeseburger", "double cheeseburger")),
    "bacon_burger": ("restaurant", ("bacon cheeseburger", "burger with bacon")),
    "veggie_burger": ("restaurant", ("veggie burger", "plant based burger")),
    "sliders": ("restaurant", ("sliders", "mini hamburger sliders")),
    # Mexicana
    "burrito_bowl": ("restaurant", ("burrito bowl", "rice bowl with meat and beans")),
    "fajitas": ("restaurant", ("fajitas", "chicken fajitas with peppers")),
    "enchiladas": ("restaurant", ("enchiladas", "enchiladas with sauce")),
    "tamales": ("restaurant", ("tamales", "corn husk tamales")),
    "guacamole": ("restaurant", ("guacamole", "guacamole with tortilla chips")),
    "refried_beans": ("restaurant", ("refried beans", "refried bean side")),
    "chili_con_carne": ("restaurant", ("chili con carne", "beef chili bowl")),
    # Italiana
    "spaghetti_meatballs": ("restaurant", ("spaghetti with meatballs", "spaghetti and meatballs")),
    "spaghetti_carbonara": ("restaurant", ("spaghetti carbonara", "pasta carbonara")),
    "penne_arrabbiata": ("restaurant", ("penne arrabbiata", "spicy penne pasta")),
    "fettuccine_alfredo": ("restaurant", ("fettuccine alfredo", "alfredo pasta")),
    "ravioli": ("restaurant", ("ravioli", "stuffed ravioli")),
    "gnocchi": ("restaurant", ("gnocchi", "potato gnocchi")),
    "risotto": ("restaurant", ("risotto", "creamy risotto")),
    "pesto_pasta": ("restaurant", ("pesto pasta", "pasta with pesto")),
    "bruschetta": ("restaurant", ("bruschetta", "toasted bread with tomatoes")),
    "calzone": ("restaurant", ("calzone", "folded pizza calzone")),
    "garlic_bread": ("restaurant", ("garlic bread", "garlic bread sticks")),
    "caprese_salad": ("restaurant", ("caprese salad", "mozzarella tomato basil salad")),
    # Asiática
    "fried_rice": ("restaurant", ("fried rice", "chinese fried rice")),
    "lo_mein": ("restaurant", ("lo mein", "chow mein noodles")),
    "chow_mein": ("restaurant", ("chow mein", "stir fried noodles")),
    "spring_rolls": ("restaurant", ("spring rolls", "fried spring rolls")),
    "dumplings": ("restaurant", ("dumplings", "steamed dumplings")),
    "dim_sum": ("restaurant", ("dim sum", "dim sum basket")),
    "kung_pao_chicken": ("restaurant", ("kung pao chicken", "spicy kung pao")),
    "sweet_sour_chicken": ("restaurant", ("sweet and sour chicken", "sweet sour pork")),
    "general_tso": ("restaurant", ("general tso chicken", "general tsos chicken")),
    "moo_shu": ("restaurant", ("moo shu pork", "moo shu wrap")),
    "wonton_soup": ("restaurant", ("wonton soup", "wonton noodle soup")),
    "hot_sour_soup": ("restaurant", ("hot and sour soup", "hot sour soup")),
    "miso_soup": ("restaurant", ("miso soup", "miso soup with tofu")),
    "ramen": ("restaurant", ("ramen", "ramen noodle bowl")),
    "udon": ("restaurant", ("udon noodles", "udon soup bowl")),
    "pho": ("restaurant", ("pho", "pho noodle soup")),
    "sushi_roll": ("restaurant", ("sushi roll", "california roll", "sushi platter")),
    "sashimi": ("restaurant", ("sashimi", "raw fish slices")),
    "tempura": ("restaurant", ("tempura", "shrimp tempura")),
    "teriyaki_chicken": ("restaurant", ("teriyaki chicken", "teriyaki bowl")),
    "pad_thai": ("restaurant", ("pad thai", "thai pad thai noodles")),
    "green_curry": ("restaurant", ("thai green curry", "green curry with rice")),
    "massaman_curry": ("restaurant", ("massaman curry", "thai massaman")),
    "chicken_tikka_masala": ("restaurant", ("chicken tikka masala", "tikka masala with rice")),
    "butter_chicken": ("restaurant", ("butter chicken", "indian butter chicken")),
    "biryani": ("restaurant", ("biryani", "chicken biryani")),
    "naan": ("restaurant", ("naan bread", "garlic naan")),
    "samosa": ("restaurant", ("samosa", "fried samosa")),
    "curry": ("restaurant", ("curry", "curry with rice")),
    "kebab": ("restaurant", ("kebab", "chicken kebab skewer")),
    "gyro": ("restaurant", ("gyro", "gyro pita wrap")),
    "falafel": ("restaurant", ("falafel", "falafel pita")),
    "hummus": ("restaurant", ("hummus", "hummus with pita")),
    # Carnes / mariscos
    "ribeye_steak": ("restaurant", ("ribeye steak", "ribeye on the plate")),
    "sirloin_steak": ("restaurant", ("sirloin steak", "grilled sirloin")),
    "filet_mignon": ("restaurant", ("filet mignon", "tenderloin steak")),
    "tbone_steak": ("restaurant", ("t bone steak", "tbone steak")),
    "pork_chop": ("restaurant", ("pork chop", "grilled pork chop")),
    "pork_ribs": ("restaurant", ("pork ribs", "bbq ribs")),
    "lamb_chops": ("restaurant", ("lamb chops", "grilled lamb chops")),
    "roast_beef": ("restaurant", ("roast beef", "roast beef slices")),
    "meatloaf": ("restaurant", ("meatloaf", "meatloaf with sauce")),
    "brisket": ("restaurant", ("brisket", "smoked brisket")),
    "short_ribs": ("restaurant", ("beef short ribs", "braised short ribs")),
    "turkey_breast": ("restaurant", ("turkey breast", "roasted turkey slices")),
    "rotisserie_chicken": ("restaurant", ("rotisserie chicken", "whole roasted chicken")),
    "chicken_parmesan": ("restaurant", ("chicken parmesan", "chicken parmigiana")),
    "chicken_cordon_bleu": ("restaurant", ("chicken cordon bleu", "stuffed chicken breast")),
    "shrimp": ("restaurant", ("shrimp", "grilled shrimp", "sauteed shrimp")),
    "shrimp_scampi": ("restaurant", ("shrimp scampi", "garlic shrimp pasta")),
    "fish_tacos": ("restaurant", ("fish tacos", "baja fish tacos")),
    "crab_cakes": ("restaurant", ("crab cakes", "crab cake patties")),
    "lobster": ("restaurant", ("lobster tail", "butter lobster")),
    "mussels": ("restaurant", ("mussels", "steamed mussels")),
    "clams": ("restaurant", ("clams", "steamed clams")),
    "oysters": ("restaurant", ("oysters", "raw oysters on shell")),
    "tuna_steak": ("restaurant", ("tuna steak", "seared tuna")),
    "cod": ("restaurant", ("cod fillet", "baked cod")),
    "tilapia": ("restaurant", ("tilapia fillet", "grilled tilapia")),
    "trout": ("restaurant", ("trout fillet", "pan fried trout")),
    # Guarniciones / vegetales
    "mashed_potatoes": ("restaurant", ("mashed potatoes", "creamy mashed potatoes")),
    "baked_potato": ("restaurant", ("baked potato", "loaded baked potato")),
    "sweet_potato": ("restaurant", ("sweet potato", "baked sweet potato")),
    "roasted_vegetables": ("restaurant", ("roasted vegetables", "roasted veggie medley")),
    "steamed_broccoli": ("restaurant", ("steamed broccoli", "broccoli florets")),
    "cauliflower": ("restaurant", ("cauliflower", "roasted cauliflower")),
    "corn_on_cob": ("restaurant", ("corn on the cob", "butter corn cob")),
    "green_beans": ("restaurant", ("green beans", "sauteed green beans")),
    "asparagus": ("restaurant", ("asparagus spears", "grilled asparagus")),
    "brussels_sprouts": ("restaurant", ("brussels sprouts", "roasted brussels sprouts")),
    "spinach": ("restaurant", ("spinach", "sauteed spinach")),
    "kale_salad": ("restaurant", ("kale salad", "kale caesar")),
    "coleslaw": ("restaurant", ("coleslaw", "cabbage slaw")),
    "potato_salad": ("restaurant", ("potato salad", "american potato salad")),
    "macaroni_salad": ("restaurant", ("macaroni salad", "pasta salad")),
    "onion_rings": ("restaurant", ("onion rings", "crispy onion rings")),
    "mozzarella_sticks": ("restaurant", ("mozzarella sticks", "fried cheese sticks")),
    # Sopas / ensaladas
    "chicken_soup": ("restaurant", ("chicken noodle soup", "chicken soup")),
    "tomato_soup": ("restaurant", ("tomato soup", "cream of tomato soup")),
    "lentil_soup": ("restaurant", ("lentil soup", "lentil stew")),
    "black_bean_soup": ("restaurant", ("black bean soup", "black bean stew")),
    "minestrone": ("restaurant", ("minestrone", "minestrone soup")),
    "chowder": ("restaurant", ("clam chowder", "corn chowder")),
    "caesar_salad": ("restaurant", ("caesar salad", "caesar salad with croutons")),
    "greek_salad": ("restaurant", ("greek salad", "greek village salad")),
    "cobb_salad": ("restaurant", ("cobb salad", "cobb salad with chicken")),
    "tuna_salad": ("restaurant", ("tuna salad", "tuna salad bowl")),
    "chicken_salad": ("restaurant", ("chicken salad", "chicken salad sandwich")),
    # Frutas / snacks
    "strawberries": ("fruit", ("strawberries", "fresh strawberry bowl")),
    "blueberries": ("fruit", ("blueberries", "fresh blueberries")),
    "grapes": ("fruit", ("grapes", "red grapes bunch")),
    "watermelon": ("fruit", ("watermelon slices", "watermelon wedge")),
    "mango": ("fruit", ("mango slices", "fresh mango")),
    "pineapple": ("fruit", ("pineapple chunks", "fresh pineapple")),
    "peach": ("fruit", ("peach", "fresh peach")),
    "pear": ("fruit", ("pear", "fresh pear")),
    "cherries": ("fruit", ("cherries", "cherry bowl")),
    "avocado": ("fruit", ("avocado", "avocado halves", "sliced avocado")),
    "nuts": ("snack", ("mixed nuts", "nuts bowl")),
    "almonds": ("snack", ("almonds", "roasted almonds")),
    "peanuts": ("snack", ("peanuts", "roasted peanuts")),
    "popcorn": ("snack", ("popcorn", "buttered popcorn")),
    "pretzels": ("snack", ("pretzels", "pretzel sticks")),
    "chips": ("snack", ("potato chips", "bag of chips")),
    "tortilla_chips": ("snack", ("tortilla chips", "corn chips")),
    "crackers": ("snack", ("crackers", "cheese crackers")),
    "trail_mix": ("snack", ("trail mix", "nuts and dried fruit mix")),
    "protein_bar": ("snack", ("protein bar", "energy bar")),
    "granola_bar": ("snack", ("granola bar", "cereal bar")),
    # Postres
    "cheesecake": ("dessert", ("cheesecake", "new york cheesecake slice")),
    "tiramisu": ("dessert", ("tiramisu", "tiramisu slice")),
    "panna_cotta": ("dessert", ("panna cotta", "italian panna cotta")),
    "cupcake": ("dessert", ("cupcake", "frosted cupcake")),
    "pie": ("dessert", ("pie", "apple pie slice")),
    "pumpkin_pie": ("dessert", ("pumpkin pie", "pumpkin pie slice")),
    "cinnamon_roll": ("dessert", ("cinnamon roll", "cinnamon bun")),
    "chocolate_cake": ("dessert", ("chocolate cake", "chocolate cake slice")),
    "carrot_cake": ("dessert", ("carrot cake", "carrot cake slice")),
    "red_velvet": ("dessert", ("red velvet cake", "red velvet slice")),
    "flan": ("dessert", ("flan", "caramel flan")),
    "pudding": ("dessert", ("pudding", "chocolate pudding cup")),
    "gelato": ("dessert", ("gelato", "gelato scoop")),
    "milkshake": ("dessert", ("milkshake", "chocolate milkshake")),
    "smoothie": ("dessert", ("fruit smoothie", "berry smoothie glass")),
    "fruit_salad": ("dessert", ("fruit salad", "mixed fruit bowl")),
    # Fast food / clásicos americanos
    "mac_and_cheese_bowl": ("fast_food", ("macaroni and cheese bowl", "creamy mac and cheese")),
    "hot_wing": ("fast_food", ("hot wings", "spicy chicken wings")),
    "chicken_burger": ("fast_food", ("chicken burger", "crispy chicken sandwich")),
    "fish_sandwich": ("fast_food", ("fish sandwich", "fried fish fillet sandwich")),
    "poutine": ("fast_food", ("poutine", "fries with gravy and cheese")),
    "loaded_fries": ("fast_food", ("loaded fries", "cheese fries")),
    "bbq_sandwich": ("fast_food", ("bbq sandwich", "pulled pork bbq")),
    "corn_salad": ("fast_food", ("corn salad", "southwest corn salad")),
    "deviled_eggs": ("fast_food", ("deviled eggs", "stuffed eggs")),
    "beef_stew": ("restaurant", ("beef stew", "hearty beef stew")),
    "chicken_stew": ("restaurant", ("chicken stew", "chicken casserole")),
    "goulash": ("restaurant", ("goulash", "hungarian goulash")),
    "paella": ("restaurant", ("paella", "seafood paella")),
    "casserole": ("restaurant", ("casserole", "baked casserole dish")),
    "quiche": ("restaurant", ("quiche", "quiche lorraine slice")),
    "shepherds_pie": ("restaurant", ("shepherds pie", "cottage pie")),
    "chicken_pot_pie": ("restaurant", ("chicken pot pie", "pot pie")),
    "cornbread": ("restaurant", ("cornbread", "southern cornbread")),
    "biscuits": ("restaurant", ("biscuits", "buttermilk biscuits")),
    "gravy": ("restaurant", ("gravy", "brown gravy bowl")),
}

NO_RELIABLE = {"soup", "cereal", "sandwich"}  # sin equivalencia defendible (mantener)


def main() -> None:
    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    curated_map = {f["canonical"]: f for f in curated["foods"]}
    source_version = curated["meta"].get("source_version", "FDC")

    foods = []
    for entry in FOOD_CATALOG:
        canonical = entry.canonical_name
        nutrition = curated_map.get(canonical)
        has_nutrition = bool(nutrition and nutrition.get("nutrition"))
        portion = REFERENCE_GRAMS.get(canonical) or REFERENCE_GRAMS.get(canonical.replace("_", " "))
        status = "NO_RELIABLE_NUTRITION" if canonical in NO_RELIABLE else (
            "PRODUCTION_READY" if has_nutrition and portion else
            "NUTRITION_READY" if has_nutrition else
            "REVIEW_REQUIRED" if canonical in ("soup", "cereal", "sandwich") else "VISUAL_ONLY"
        )
        foods.append({
            "canonical_name": canonical,
            "aliases": list(entry.clip_candidates),
            "clip_candidates": list(entry.clip_candidates),
            "category": entry.category,
            "visual_sources": [],
            "nutrition": {
                "fdc_id": nutrition.get("fdc_id"), "fdc_name": nutrition.get("fdc_name"),
                "data_type": nutrition.get("data_type"), "mapping_status": nutrition.get("mapping_status"),
                "mapping_confidence": nutrition.get("mapping_confidence"),
                "source": "USDA FoodData Central", "source_version": source_version,
                "source_id": nutrition.get("fdc_id"),
            } if has_nutrition else None,
            "portion": {
                "reference_grams": portion,
                "source": "FDC foodPortions / USDA SR Legacy",
            } if portion else None,
            "status": status,
        })

    # Expansión: VISUAL_ONLY por defecto (los mappings USDA se añaden en P4)
    for canonical, (category, candidates) in CATALOG_EXPANSION.items():
        foods.append({
            "canonical_name": canonical,
            "aliases": list(candidates),
            "clip_candidates": list(candidates),
            "category": category,
            "visual_sources": [],
            "nutrition": None,
            "portion": None,
            "status": "VISUAL_ONLY",
        })

    master = {
        "schema_version": 1,
        "description": "Food Master Catalog — separa identidad visual, nutrición USDA y porción.",
        "generated_by": "scripts/build_food_master.py",
        "foods": sorted(foods, key=lambda f: f["canonical_name"]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    statuses = Counter(f["status"] for f in master["foods"])
    print(f"TOTAL: {len(master['foods'])} clases")
    for s, n in statuses.most_common():
        print(f"  {s}: {n}")


if __name__ == "__main__":
    main()