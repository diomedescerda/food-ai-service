# Datasets — Food AI

**Estado: FASE 0 — sin datasets todavía.**

## Plan (FASE 10-12)

Estructura futura:

```
datasets/
  food-detection/        # anotaciones YOLO (boxes)
  food-segmentation/     # máscaras
  food-classification/   # imágenes por clase
```

Cada dataset documenta: origen, licencia, clases, cantidad de imágenes, split train/validation/test, formato de anotación.

Reglas:
- Comenzar con 30–50 clases colombianas (arroz, arepa, frijoles, lentejas, yuca, papa, plátano, patacón, pollo, carne, cerdo, pescado, huevo, aguacate, queso, empanada, sancocho, ajiaco, bandeja paisa, arroz con pollo, arroz de coco…).
- NUNCA introducir datasets cuya licencia no permita el uso previsto.
- Imágenes y anotaciones NO van a git (`.gitignore`).
- Scripts: `prepare_dataset.py`, `validate_dataset.py`, `split_dataset.py`.
- Correcciones de usuario (FASE 15) alimentan un dataset real tras proceso de validación.