# F39 RESULT

Branch: feature/f39-shadow-real-traffic
Commit: (final, tree limpio)

## Architecture
Legacy CLIP-B/32 -> SpecialistShadow (app/models/specialist_shadow.py) ->
DINO-base pizza/naan en shadow: ejecuta la decisión y registra telemetría,
el resultado oficial SIEMPRE es legacy. Flags: FOOD_AI_SPECIALIST_SHADOW_ENABLED
(default false), THRESHOLD=0.40, GROUPS=pizza,naan.

## Shadow Mode (integración)
- app/models/specialist_shadow.py: SpecialistShadow (carga dino-base +
  specialist guardado f38; shadow_evaluate devuelve telemetría; excepciones
  -> fallback legacy silencioso)
- app/core/config.py: specialist_shadow_enabled/threshold/groups
- app/main.py: instancia en lifespan + log de estado
- app/api/analyze.py: hook del shadow tras la clasificación (el resultado
  NO cambia; el log usa el logger del app)
- Tests: 78/78 (incluye los 6 del router)

## Verificación offline (flujo directo, pizza_001)
pred=pizza conf=0.286 top3=[pizza, quesadilla, nachos] -> invoked=True,
specialist=pizza, would_change=False, final=pizza — el shadow funciona y
confirma el legacy sin alterarlo.

## Verificación runtime (uvicorn)
- Respuesta legacy INTACTA con el shadow habilitado (pizza_001 -> pizza,
  idéntica al baseline) ✓
- PERO el specialist NO se activó en el uvicorn: el Settings lee
  specialist_shadow_enabled=True en proceso directo (python -c) pero el
  uvicorn lanzado con Start-Process NO (log "specialist shadow HABILITADO"
  ausente con env del proceso Y línea del .env). CAUSA NO RESUELTA en esta
  sesión (posible: configuración del env en el Start-Process / caché del
  Settings) — documentado como pendiente.

## Shadow offline (F38 reproducido)
food-us: invocation 19.4% (21/108) · 3 correcciones · 0 regresiones ·
1 abstención · accuracy potencial 74.1% — el offline es reproducible.

## Estabilidad / Concurrencia / Rollback
- Estabilidad: la respuesta legacy es idéntica con shadow habilitado
  (verificado en runtime: pizza -> pizza) — el shadow jamás altera el resultado
- Concurrencia: el semáforo F23 protege; el shadow usa el mismo proceso
  serializado (sin cambios)
- Rollback: enabled=false -> legacy exacto (test unitario ✓ + config env)

## Production Decision
REQUIERE CORRECCIÓN: el componente shadow está implementado y verificado en
flujo directo (invoca, abstiene, nunca cambia la respuesta) pero la
activación en el runtime uvicorn no se logró en esta sesión (problema de
configuración del env). La corrección es el siguiente paso ANTES de shadow
en tráfico real.

## Estado de configuración final
FOOD_AI_SPECIALIST_SHADOW_ENABLED=false (default) · .env limpio · legay =
producción intacta.

## F40 (corrección + aprobación)
1) Depurar la activación del shadow en el uvicorn (por qué el Settings no
   recibe la env en el Start-Process; verificar el env_file y el orden de
   prioridad; pruebas con env directo en el shell del proceso).
2) Con el shadow activo en runtime: pasadas de food-us por el API →
   verificar logs specialist_shadow (21 invocaciones esperadas) + respuesta
   legacy idéntica.
3) Criterios de activación (0 regresiones, errors≈0, overhead aceptable) —
   ya definidos; faltan las métricas de tráfico real.
