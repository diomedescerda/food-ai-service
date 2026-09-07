# F48 RESULT — General Reranker para 5.761 alimentos

Branch: feature/f48-general-reranker · Tests: 104/104 (7 nuevos) · Tree limpio

## Features (precomputadas por imagen, sin duplicar cálculos)
- retrieval_max / retrieval_mean / retrieval_rank (del Top-50 del CLIP)
- support_count (entries del canonical — ESTRUCTURAL, independiente del CLIP)
- alias_count (aliases del canonical — ESTRUCTURAL; proxy del canónico
  genérico del producto: pizza/hamburger tienen 20+ aliases)
- specialist DINO pizza/naan (solo si eligible, gate F45 intacto: conf>=0.75)
- prototype: NO disponible (los prototipos F27-F29 viven en la rama del otro
  agente) — feature ausente, sin penalización
- legacy CLIP: EXCLUIDO (F43 demostró correlación con el retrieval)

## Configs evaluadas (pesos simples, sin grid)
| Config | pesos | food-us R@1 |
|---|---|---|
| base_max (F47) | 1.0*max | 10.2% |
| A retr+support | max + 0.5*support_norm | 10.2% (support no mueve: la mayoría de candidatos tienen support 1) |
| B +alias | A + 0.1*alias_norm | 38.0% |
| D +specialist | A + 1.0*dino | 20.4% |
| E todo | B + specialist | 38.9% |

## Comparación F47 → F48 (config E)
| Dataset | F47 R@1 | F48 R@1 | Δ | F47 R@5 | F48 R@5 | R@10 | R@20 | R@50 |
|---|---|---|---|---|---|---|---|---|
| food-us | 10.2% | **38.9%** | +28.7 | 22.2 | 53.7 | 53.7 | 53.7 | 53.7 |
| v1 | 8.7% | **21.0%** | +12.3 | 16.9 | 31.0 | 31.9 | 31.9 | 31.9 |
| Food-101 | 9.5% | **34.2%** | +24.7 | 26.0 | 49.6 | 50.6 | 51.3 | 51.6 |

El R@1 recupera el nivel del catálogo 1.451 (food-us 38.9 = exacto; v1 21.0
> 19.0; Food-101 34.2 > 32.2 — SUPERADO) con el catálogo 4x. El R@50 no
cambia (el reranker no puede añadir recall — solo ordena).

## Ranking (config E)
| Dataset | MRR | med_rank GT | promoted | demoted |
|---|---|---|---|---|
| food-us | 0.458 (base 0.185) | 1 (base 6) | 43 | 4 |
| v1 | 0.253 (base 0.115) | 1 (base 6) | 58 | 7 |
| Food-101 | 0.415 (base 0.180) | 1 (base 5) | 1414 | 105 (2.7%) |

Las demotions (food101 105): canónicos genéricos que desplazan el GT —
compensadas por el +24.7 de R@1 (los R@5+ suben en los 3 datasets).

## Error buckets (food-us)
A (GT fuera del Top-50 — NO recuperable por el reranker): 50/108 (46%)
B (GT en Top-50 rank>1 — EL problema de F48): 47/108 (44%) → el E rescata
  42/47 (89%)
C (variante semántica cerca): 0

## Specialista
Integrado con la regla F45 intacta (dino_base, pizza/naan, thr 0.75, gate
top-3, conf<0.40). Config D: +10.2 pts solo (food-us); E: +0.9 sobre B. No
se crearon especialistas nuevos.

## Performance
Retrieval ~80ms + grouping <1ms + rerank <1ms (numpy puro, sin modelo
adicional). Memory: sin cambio (~12MB índice). El DINO solo en eligible.

## Producción
Intacta: retrieval off, shadow off, specialist off, legacy = respuesta,
104/104.

## Decisión
APPROVE F48 — el reranker determinista (retrieval + support + alias +
specialist) recupera el Top-1 al nivel del catálogo 1.451 con el catálogo
4x (food-us 38.9, Food-101 34.2 — superado) sin entrenar nada. El cuello
restante: el bucket A (46% del food-us: GT fuera del Top-50 — el recall
del retrieval CLIP con 5.761). El alias_count (proxy del canónico genérico)
es la señal clave; el support aporta poco; el specialist aporta en
pizza/naan (regla F45 intacta).

## Siguiente
1) El bucket A (recall): retrieval con embeddings multi-idioma o el
   canonical más el alias en el índice (las aliases de los canónicos
   genéricos como prompts adicionales).
2) F49: nutrition mapping masivo.
3) F50: unknown/confidence + rollout.
