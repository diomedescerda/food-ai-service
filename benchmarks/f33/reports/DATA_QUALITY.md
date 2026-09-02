# DATA QUALITY — F33

| Clase | Candidatas | VALID | AMBIGUOUS | WRONG | DUPLICATE | Train final |
|---|---|---|---|---|---|---|
| chicken_nuggets | 32 | 32 | 0 | 0 | 0 | 32 |
| french_fries | 32 | 0 | 0 | 32 | 0 | 0 |
| fried_chicken | 32 | 32 | 0 | 0 | 0 | 32 |
| hamburger | 32 | 32 | 0 | 0 | 0 | 32 |
| naan | 24 | 1 | 0 | 0 | 23 | 1 |
| nachos | 32 | 0 | 32 | 0 | 0 | 0 |
| pizza | 32 | 32 | 0 | 0 | 0 | 32 |
| quesadilla | 32 | 32 | 0 | 0 | 0 | 32 |
| rice | 32 | 32 | 0 | 0 | 0 | 32 |
| sandwich | 32 | 3 | 0 | 0 | 29 | 3 |
| taco | 32 | 0 | 0 | 0 | 32 | 0 |
| TOTAL | 344 | 145 | 58 | 41 | 100 | 129 |

Hallazgos: fries (las 32 imágenes de Commons NO matchean el legacy top-1 —
eliminadas); naan/sandwich/taco con descargas duplicadas de Commons (el dedup
0.97 las elimina); nachos ambiguas (top-5 del legacy). La validación estricta
elimina exactamente las clases de confusión — el umbral top-1 es la elección
crítica.
