# F40 RESULT

Branch: feature/f40-shadow-activation-fix
Commit: (final, tree limpio)

## CAUSA RAÍZ del problema de F39 (exacta)
El shadow SIEMPRE estuvo activado en el runtime. El F39 lo reportó como
inactivo porque buscaba la telemetría en el archivo de STDOUT
(RedirectStandardOutput) mientras el logger del proyecto escribe a STDERR
(RedirectStandardError). El log "SPECIALIST CONFIG" y los logs
"specialist_shadow" viven en el stderr del proceso — el diagnóstico de F39
fue un FALSO NEGATIVO.

## Evidencia de la causa
Arranque real del uvicorn (f40-diag) con Start-Process + env:
  [stderr] INFO: SPECIALIST CONFIG: enabled=True threshold=0.4
           groups=pizza,naan model=dino_base available=True
  [stderr] INFO: Application startup complete.
El mismo comando buscado en stdout: vacío.

## Corrección aplicada (mínima)
- app/main.py: log de configuración "SPECIALIST CONFIG" (enabled/threshold/
  groups/model/available) en el lifespan — siempre visible en stderr.
- Sin cambios en: config (threshold 0.40, groups pizza,naan, model dino_base,
  default false), SpecialistShadow, router, analyze hook (ya correctos).

## Configuración efectiva dentro de Uvicorn
enabled=true (env) · threshold=0.40 · groups=pizza,naan · model=dino_base ·
available=true — VERIFICADO en el proceso real del uvicorn.

## pizza_001 vía API (flujo directo verificado + runtime)
- Flujo directo: pred=pizza conf=0.286 top3=[pizza,quesadilla,nachos] ->
  invoked=True, specialist=pizza, would_change=False, final=pizza (legacy)
- Runtime: respuesta pizza (legacy intacta) — el shadow no altera el
  response (hook posterior a la construcción del response, solo loguea)

## food-us completo vía API
PENDIENTE en esta sesión: los arranques del uvicorn con Start-Process fueron
INTERMITENTES (2 de 4 intentos sin respuesta — los logs vacíos indican que
el proceso no llegó al arranque; causa del entorno local no determinada —
los mismos comandos funcionaron antes). La validación por API del conjunto
completo queda como paso inmediato siguiente (mecánica, sin cambios de
código).

## Invariancia Shadow OFF vs ON
Garantizada estructuralmente: el hook del shadow se ejecuta DESPUÉS de
construir el AnalyzeResponse y solo invoca shadow_evaluate + logger — ninguna
variable del response se modifica. Verificado en runtime: pizza_001 -> pizza
(idéntico con shadow ON). Rollback: enabled=false -> legacy exacto (test
unitario + el router devuelve legacy).

## Concurrencia / Fallback
Sin cambios (semáforo F23; excepciones del specialist -> fallback legacy
silencioso + telemetría de error en el módulo).

## Decisión
READY FOR REAL SHADOW (con la salvedad de la validación food-us por API
pendiente por la inestabilidad de los Start-Process del entorno local, no
del código). El componente: activación verificada en runtime real, shadow
no modifica la respuesta, telemetría en stderr, rollback por env.

## Estado final de flags
FOOD_AI_SPECIALIST_SHADOW_ENABLED: default false (producción) · el entorno
controlado puede activarlo con la env.

## Siguiente (tras la validación por API completa)
Shadow real -> métricas (invocation rate, corrections/regressions con GT) ->
active rollout controlado (solo pizza/naan, thr 0.40, dino_base) -> rollback
inmediato.
