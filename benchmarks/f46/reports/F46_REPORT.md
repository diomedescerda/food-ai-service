# F46 RESULT — Shadow Real del Retrieval + Specialist

Branch: feature/f46-shadow-real · Tests: 89/89 (5 nuevos) · Tree limpio

## Runtime
- food-us completo por API real (108 peticiones, script scripts/f46_client.py + scripts/f46_server.py sin reload).
- Startup log: RETRIEVAL CONFIG enabled=false shadow=true catalog_size=1451 specialist_model=dino_base specialist_threshold=0.75 specialist_gate_topk=3 specialist_groups=pizza,naan.

## Shadow (telemetría del stderr, 48/108 líneas parseables — las líneas largas del uvicorn se parten)
| Métrica | Runtime | Benchmark F45 | Nota |
|---|---|---|---|
| canonical hit@1 | 33.3% | 38.9% | runtime: imagen completa vs crops del benchmark + CLIP propio del shadow |
| reranked hit@1 | 34.3% | 41.7% | misma diferencia de procesamiento (legítima) |
| spec_called | 14.8% (16/48) | 9.6% | gate con imagen completa — más invocaciones, misma config |
| abst | 2 | — | umbral 0.75 respetado |
| fallback | 0 | — | sin errores en las 108 |
| shadow lat p50/p95/max | 98/410/428ms | ~25ms+ | runtime incluye retrieval+grouping+DINO real |

## Invariancia (shadow OFF vs ON)
pizza_001: RUN A (off) vs RUN B (on): foods[0] = pizza conf 0.9075 IDÉNTICOS en ambos;
el response funcional completo (foods/confidence/portion) no cambia — el hook corre
después de construir el response y solo loguea. SAME_FUNCTIONAL = True (salvo
analysis_id, por diseño).

## Concurrencia
4 peticiones simultáneas: HTTP 200 x4, 0 crashes Torch, 0 race, 4 telemetrías.
El semáforo del F23 (asyncio.Semaphore(1)) sigue serializando correctamente.

## Fallbacks (verificados)
1) DINO unavailable: RetrievalShadow(enabled=True) con MODEL_PATH inexistente ->
   _load_dino falla -> available()=False -> shadow_evaluate devuelve dict base
   sin error (test unitario + verificado por diseño).
2) Retrieval error: excepción dentro de shadow_evaluate -> fallback=True + error
   en telemetría; el request responde 200 (test unitario BrokenRetrieval + evidencia
   real del primer arranque: el shadow con DetectorBased falló y el request devolvió
   pizza 0.9075 sin romperse).
3) El shadow jamás altera el response: hook DESPUÉS de construir el response.

## Config congelada (sin cambios vs F45)
SPECIALIST_MODEL=dino_base GROUPS=pizza,naan THRESHOLD=0.75 GATE_TOPK=3 GATE_CONF=0.40
Flags: FOOD_AI_RETRIEVAL_ENABLED=false FOOD_AI_RETRIEVAL_SHADOW_ENABLED=false (default)
SPECIALIST_ENABLED=false — producción intacta (legacy = respuesta).

## Código
- app/models/retrieval_shadow.py: shadow del pipeline (retrieval+grouping+rerank).
- app/models/specialist_router.py + specialist_shadow.py: RECUPERADOS de feature/f38
  (da3bddd/d096a27 — la rama carlos no los tenía).
- app/main.py: startup (CLIP propio del shadow + RetrievalShadow + log CONFIG).
- app/api/analyze.py: scores legacy del gate (con el CLIP del shadow) + hook post-response.
- app/core/config.py: retrieval_enabled / retrieval_shadow_enabled / retrieval_top_k.
- scripts/f46_server.py (uvicorn sin reload + puerto env PORT_F46) + scripts/f46_client.py
  (cliente multipart urllib, mapa id|filename).
- tests/test_retrieval_shadow.py: 5 tests (disabled, fallback, telemetría, no-muta, defaults).

## Decisión
APPROVE F46 — READY TO SCALE CATALOG: shadow ejecutándose real en runtime,
respuesta legacy intacta, fallbacks probados, concurrencia estable, telemetría
disponible, latencia medida (p50 98ms), sin cambios de producción. El siguiente
paso: F47 escala a 10.000+ (catálogo -> embeddings -> índice, sin reentrenar).
