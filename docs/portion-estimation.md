# Estimación de porción (FASE 6-7)

**Advertencia**: `estimatedGrams != measuredGrams`. Sin cámara calibrada ni
escala física, una foto NO proporciona el peso real. El resultado es una
estimación aproximada, explícitamente comunicada como tal.

## FASE 7 — Advanced (depth): veredicto experimental

**¿La profundidad mejora de forma medible la estimación de porciones?**

**Evidencia insuficiente / NO recomendado actualmente para gramos:**

| Métrica | Valor |
|---|---|
| Depth load (única vez) | 13.2 s |
| Depth inference (por imagen) | 756–863 ms |
| Basic portion | ~0 ms |
| Señal del depth (contraste alimento vs fondo) | banana 0.770 · apple 0.747 · pizza 0.119 |
| Gramos producidos por Advanced | NINGUNO (sin escala física) |

El depth map funciona y discrimina profundidad relativa real (banana/apple
significativamente más cercanos que el fondo), pero **la profundidad monocular
es relativa**: sin una referencia de tamaño conocido (plato/vaso/cubierto) o
calibración, volumen relativo ≠ masa. El AdvancedPortionEstimator devuelve
`method="advanced_depth_relative"` con gramos nulos (confidence 0) — nunca
finge masa.

**Comparación Basic vs Advanced**: Basic produce gramos (referencia FDC ±20 %);
Advanced no produce gramos sin escala → no existe MAE/RMSE comparable y no se
puede afirmar mejora. La geometría (mediana, percentiles, contraste) queda
disponible como característica para futuras fases con escala física o dataset
con ground truth.

## Algoritmo básico (`BasicPortionEstimator`)

```
1. Referencia: gramos de UNA porción doméstica típica del alimento (USDA FDC)
2. Tamaño visual: área relativa de la máscara (mask_area / image_area)
   fallback: área del bbox si no hay máscara
3. Gramos: estimated = referencia × factor(tamaño); rango [min, max]
4. Confidence: 0.55 fija (heurística visual simple); UNKNOWN → 0
```

Paso 2 **no** afirma que píxeles = gramos: el área relativa solo discrimina
small/medium/large (sin escala física).

## Referencias (USDA FoodData Central, medidas domésticas)

| Alias | 1 porción doméstica | Gramos |
|---|---|---|
| banana | 1 banana mediana (7"-7⅞") | 118 |
| apple | 1 manzana mediana | 182 |
| orange | 1 naranja mediana | 131 |
| broccoli | 1 taza cruda picada | 91 |
| carrot | 1 zanahoria mediana | 61 |
| pizza | 1 rebanada (⅛ de pizza 12") | 107 |
| hot dog | 1 frankfurter | 57 |
| donut | 1 dona | 60 |
| cake | 1 rebanada (1/12 de pastel) | 95 |

Sin referencia (p. ej. sandwich) → **UNKNOWN**, nunca se inventa un número.

## Tamaños y gramos

| Tamaño | Criterio visual (área relativa) | Factor | Rango |
|---|---|---|---|
| small | ≤ 0.12 | ×0.8 | [0.6, 0.9] × ref |
| medium | 0.12–0.30 | ×1.0 | [0.8, 1.2] × ref |
| large | ≥ 0.30 | ×1.2 | [1.1, 1.5] × ref |
| unknown | sin referencia o sin área | — | — |

Confidence 0.55 (heurística básica; no se combina con la confianza de
detección). La estimación añade ~0 ms de latencia (sin modelo).

## Resultado real (assets de prueba)

| Imagen | Área relativa | Tamaño | Estimated | Rango |
|---|---|---|---|---|
| pizza (330×247) | 0.55 | large | 128 g | 118–160 g |
| banana (330×291) | 0.26 | medium | 118 g | 94–142 g |
| apple (330×299) | 0.31 | large | 218 g | 200–273 g |

## Limitaciones

- Sin profundidad/volumen/densidad → el tamaño visual no distingue plato
  lleno vs cámara cerca (FASE 7 abordará con métricas previas).
- El área relativa depende del encuadre y distancia de la cámara.
- Confidence fija: no refleja cuán extremo es el alimento en la imagen.
- No es una medición: destinada a educación/estimación, no a precisión clínica.

## Estructura

`IPortionEstimator` (app/models/portion_base.py) → `BasicPortionEstimator`
(app/models/basic_portion_estimator.py), separado del detector/segmentador/
clasificador/proveedor de nutrición. Sustituible por el estimador avanzado
(FASE 7) sin tocar analyze.py.