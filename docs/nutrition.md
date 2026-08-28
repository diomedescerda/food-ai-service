# Nutrición — Food AI

**Estado: FASE 5 — base nutricional implementada (PostgreSQL, schema `foodai.`).**

## Fuente de datos

- **Fuente**: USDA FoodData Central (fdc.nal.usda.gov) — base oficial de referencia del gobierno de EE. UU.
- **Licencia**: datos de dominio público (gobierno de EE. UU.), sin restricciones de uso — incorporable legalmente.
- **Versión**: valores tomados manualmente de entradas FDC (consulta 2026-08-27). `SourceVersion = "2026-08-27"` por fila.
- **Importación**: seeder EF idempotente (`FoodAiNutritionSeeder`, corre al arrancar la API). Reproducible desde entorno limpio.

## Unidad canónica

Todos los valores están expresados **por 100 g** (`ServingGrams = 100`). El cálculo de gramos finales llegará en FASE 6-8 (nunca la IA escribe valores nutricionales).

## Alimentos sembrados (alias YOLO → entrada FDC)

| Alias YOLO | Entrada FDC | Categoría | kcal | Prot | Carb | Fat | Fibra | Azúcar | Na (mg) |
|---|---|---|---|---|---|---|---|---|---|
| banana | Banana, raw | fruit | 89 | 1.09 | 22.84 | 0.33 | 2.6 | 12.23 | 1 |
| apple | Apple, raw, with skin | fruit | 52 | 0.26 | 13.81 | 0.17 | 2.4 | 10.39 | 1 |
| orange | Orange, raw | fruit | 47 | 0.94 | 11.75 | 0.12 | 2.4 | 9.35 | 0 |
| broccoli | Broccoli, raw | vegetable | 34 | 2.82 | 6.64 | 0.37 | 2.6 | 1.7 | 33 |
| carrot | Carrot, raw | vegetable | 41 | 0.93 | 9.58 | 0.24 | 2.8 | 4.74 | 69 |
| pizza | Pizza, cheese, per 100 g | prepared | 266 | 11.39 | 33.33 | 10.4 | 2.3 | 3.6 | 598 |
| hot dog | Frankfurter, beef, per 100 g | prepared | 290 | 12.0 | 2.7 | 25.0 | 0 | 1.1 | 1050 |
| donut | Doughnuts, cake-type, plain | confection | 452 | 4.9 | 51.3 | 25.4 | 1.5 | 24.8 | 445 |
| cake | Cake, yellow, plain, no frosting | confection | 361 | 5.3 | 53.2 | 14.6 | 0.7 | 27.9 | 392 |

## Limitaciones honestas

- **`sandwich` SIN entrada**: el modelo YOLO lo detecta, pero no existe entrada FDC fiable para un "sándwich genérico" (pan+relleno variable). No se inventan equivalencias → `FOOD_NOT_FOUND`. Requiere definición de referencia propia (decisión futura).
- **hot dog**: entrada del frankfurter SOLO (sin pan) — documentado para no asumir la preparación completa.
- **pizza**: referencia de pizza de queso; una pizza pepperoni difiere. Fine-tuning del catálogo en fases futuras.

## Preparación para comida colombiana

El modelo permite: `Foods` + `FoodNutrition` + `FoodAliases` → añadir arroz blanco, arepa, frijoles, patacón, yuca, sancocho, ajiaco, bandeja paisa... sin cambios de esquema. La detección de esas clases llega con FASE 10-12.

## Esquema (schema `foodai.`)

```
foods           (id, name, display_name, category, is_active, created_at, updated_at)
food_nutrition  (id, food_id FK, serving_grams=100, calories, protein, carbohydrates, fat,
                 fiber, sugar, sodium — todos numeric(10,2) —, source, source_version, imported_at)
food_aliases    (id, food_id FK, alias UNIQUE, source)
```

Decimales (`numeric(10,2)`), nunca float. Migración `AddFoodAiNutrition` + GRANT a `app_user`.

## Servicio

`INutritionProvider` → `DatabaseNutritionProvider` (Application/Infrastructure): alias o nombre canónico → entrada más reciente (`ImportedAt`). Endpoint de verificación: `GET /api/v1/foodai/nutrition/{foodKey}` (404 controlado).