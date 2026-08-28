# Dataset strategy — US Food (FASE 10)

## Objetivo

Construir el dataset para evolucionar de YOLO11n-COCO a un modelo de comida
**consumida en Estados Unidos**: restaurantes, fast food, casual dining, home
meals, delivery/takeout, supermercados. NO es un dataset colombiano.

## Problema actual y limitaciones COCO

- YOLO11n COCO solo tiene 10 clases de comida (banana, apple, sandwich,
  orange, broccoli, carrot, hot dog, pizza, donut, cake).
- Faltan las clases de mayor consumo en USA: hamburger, french fries,
  fried chicken, salad, pasta, steak, rice, pancakes, tacos, burrito, etc.
- COCO no distingue platos compuestos ni variantes (burger vs cheeseburger).

## Platos vs componentes (decisión clave)

**Decisión**: para el MVP se detectan **PLATOS COMPLETOS** (hamburger entero,
pizza entera, hot dog completo), no sus ingredientes (bun, patty, cheese).

- Alternativas: (a) componentes por separado, (b) plato completo.
- Motivo: la detección por componentes desde una foto única es poco fiable
  (solapamiento, ocultación); la Nutrition DB (USDA FDC) tiene entradas
  "as consumed" para platos completos (p. ej. "Hamburger, single patty"),
  lo que alinea detección → porción → nutrición.
- Impacto: la FASE 12 decidirá si el modelo especializado añade componentes
  (ej. taco: tortilla/relleno) sin eliminar platos.

## Taxonomía (metadata; no son clases YOLO por sí solas)

```
food
 ├── fast_food:      hamburger, french_fries, hot_dog, fried_chicken, nuggets
 ├── restaurant:     pizza, pasta, steak, salad, tacos, burrito, rice
 ├── breakfast:      pancakes, waffles, eggs, bacon
 ├── dessert:        donut, cake, ice_cream, cookies, brownie
 ├── fruit:          apple, banana, orange
 ├── vegetable:      broccoli, carrot
 └── snack:          nachos, chips
```

Cada imagen lleva `category` en metadata — no requiere clasificación jerárquica
en el modelo.

## Clases MVP (prioridad)

| Clase | Prio | Justificación |
|---|---|---|
| pizza | HIGH | consumo masivo; COCO ✓; FDC ✓ |
| hamburger | HIGH | icónica fast food; FDC ✓ |
| french_fries | HIGH | acompañante universal; FDC ✓ |
| hot_dog | HIGH | COCO ✓; FDC ✓ |
| sandwich | HIGH | COCO ✓; gap FDC (resolver con referencia propia) |
| fried_chicken | HIGH | consumo alto; FDC ✓ |
| salad | HIGH | restaurante/casero; FDC ✓ |
| pasta | HIGH | FDC ✓ (spaghetti cooked) |
| steak | HIGH | FDC ✓ |
| rice | HIGH | básico universal; FDC ✓ |
| pancakes | MEDIUM→HIGH | desayuno; FDC ✓ |
| tacos | MEDIUM | variedad de cocinas en USA; FDC ✓ |

Descartadas del MVP: nachos, burrito, eggs, bacon, ice_cream, cookies,
brownie, waffles, mac_and_cheese, mashed_potatoes, sub, bagel, toast
(se añadirán en iteraciones posteriores del dataset).

## Fuentes

- **Wikimedia Commons** (API MediaWiki): única fuente del MVP.
  - Ventaja: trazabilidad completa (URL + licencia por imagen), automatizable.
  - Licencias aceptadas: CC0, CC BY, CC BY-SA, Public Domain.
  - Excluidas: CC BY-NC/ND, GFDL-ambiguas, imágenes sin licencia clara.
- Food-101: evaluado y **NO usado** (licencia académica/no comercial explícita
  para redistribution — incompatible con el objetivo).
- Open Images v7: anotaciones CC-BY 4.0 + imágenes Flickr CC-BY 2.0 — viable
  como fuente futura con atribución; queda documentado como pendiente.
- Feedback de usuarios (FASE 9): candidato futuro SOLO tras curación y
  consentimiento — nunca automático.

## Estructura

```
datasets/food-us-v0.1/
├── images/{train,val,test}/      # JPG ~640px, nombre {class}_{id}.jpg
├── labels/{train,val,test}/      # vacío — FASE 11 (YOLO txt)
├── metadata/
│   ├── images.csv                # image_id, filename, class, split, source_url,
│   │                             #   license, license_url, category, downloaded_at
│   ├── sources.csv               # source, license, attribution, url
│   └── classes.csv               # class, priority, category, target_count
├── splits.json                   # semilla y proporción
└── dataset.yaml                  # definición oficial de clases (ids versionados)
```

## Versionado

- `food-us-v0.1` — primera versión MVP (metadata + imágenes; anotaciones en
  FASE 11). `dataset.yaml` congela el orden de clases; los ids NO se
  reordenan en versiones futuras sin migración.

## Splits y leakage

- Seed fija 42, proporción 70/20/10 por clase.
- Deduplicación previa al split: hash MD5 (duplicados exactos) + pHash
  (casi-duplicados, distancia Hamming ≤ 8) sobre TODO el dataset → nunca la
  misma foto en train y val/test.
- Límite conocido: imágenes de la misma sesión/fuente no siempre detectables
  sin metadata externa; documentado.

## Ground truth

- `prediction` (modelo) ≠ `annotation` (anotador, FASE 11) ≠ `feedback`
  (usuario, FASE 9). Solo las annotations serán ground truth de entrenamiento.
- Feedback → candidate sample SOLO tras validación manual (curación).

## Porciones / pesos

- Sin pesos reales en el MVP (no se inventan).
- Metadata incluirá `portion_variation` (small/medium/large) cuando la imagen
  permita anotarlo en FASE 11.
- Objetivo futuro: imágenes con `actual_weight_grams` + `measurement_method`
  (registradas, nunca inventadas).

## Calidad

- Filtros de build: imagen legible (PIL), ancho ≥ 200 px, sin duplicados.
- Casos difíciles (occlusion, múltiples alimentos, platos de papel, takeout,
  diferentes luces/fondos) se conservan: son deseables para robustez.
- `bad data` (ilegible/corrupta) se excluye; `hard example` se conserva.