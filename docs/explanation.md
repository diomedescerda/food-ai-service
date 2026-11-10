# Explicación fundamentada y endurecimiento de FoodAI

El servicio no decide objetivos nutricionales ni prescribe cambios. El backend
autenticado obtiene el plan aprobado actual, calcula las diferencias y envía
solo los datos confirmados. Sin plan aprobado, el backend responde
`availability: "no_approved_plan"` sin llamar al servicio. Guardar un intake
no depende de este endpoint.

## Contrato interno v1

`POST /internal/foodai/explain`, `Content-Type: application/json`,
`X-FoodAI-Service-Key: <secreto compartido>`. Los modelos rechazan campos
adicionales, conversiones de tipos, infinitos y NaN. `request_id` es un UUID.
Todos los IDs de evidencia son únicos dentro de la petición y cumplen
`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$`. No se envían identidad, fotografías,
historial de conversación ni memoria de chat.

```json
{
  "schema_version": "1",
  "request_id": "3f3f0f0f-1111-2222-3333-444444444444",
  "locale": "es",
  "plan": {
    "id": "plan-1",
    "version": "approved-1",
    "targets": [{"id": "target-protein", "metric": "protein_g", "amount": 100, "unit": "g"}]
  },
  "diffs": [{"id": "diff-protein", "target_id": "target-protein", "metric": "protein_g", "actual": 20, "target": 100, "delta": -80, "unit": "g"}],
  "confirmed_foods": [{"id": "food-lentils", "name": "Arroz con lentejas", "grams": 200}],
  "restrictions": [{"id": "restriction-1", "label": "Maní"}],
  "missing_values": []
}
```

Métricas permitidas: `calories`, `protein_g`, `carbs_g`, `fat_g`, `fiber_g`.
La unidad de calorías es `kcal`; las demás usan `g`. Máximos: cinco objetivos,
cinco diferencias (un objetivo/diferencia por métrica), veinte alimentos,
veinte restricciones, treinta valores faltantes. Nombres y etiquetas tienen
1–200 caracteres. Objetivos/actuales tienen rango 0–1 000 000; gramos confirmados
son positivos y no superan 100 000. Cada diferencia debe señalar un objetivo
existente, con la misma métrica, unidad y cantidad; `delta = actual - target`
con tolerancia absoluta 0.01 para redondeo.

Para totales desconocidos o parciales, `actual` y `delta` deben ser `null`.
No producen comparaciones numéricas. Un valor faltante cuyo `field` coincide
con la métrica, o es `nutrition`/`totals`, también impide esa comparación.

```json
{
  "schema_version": "1",
  "request_id": "3f3f0f0f-1111-2222-3333-444444444444",
  "locale": "es",
  "observations": [{
    "id": "metric_diff:diff-protein",
    "claim": "metric_diff",
    "evidence_ids": ["diff-protein", "target-protein"],
    "text": "Tus alimentos confirmados aportan 20 g de proteína; tu objetivo aprobado es de 100 g. Son 80 g menos que ese objetivo."
  }],
  "questions": [{
    "id": "review_restriction:restriction-1", "claim": "review_restriction",
    "evidence_ids": ["restriction-1"], "text": "Ten presente esta restricción: Maní. Revisa los ingredientes y la preparación; una foto no confirma su ausencia."
  }],
  "suggestions": [{
    "id": "review_plan:diff-protein", "claim": "review_plan",
    "evidence_ids": ["diff-protein", "target-protein"],
    "text": "Si tu plan aprobado incluye una opción de proteína para esta comida, comprueba que la hayas registrado y confirma su porción antes de ajustar el registro."
  }],
  "provenance": {
    "provider": "deterministic", "vendor": "local", "model": "evidence-templates-v1",
    "schema_version": "1", "prompt_version": "foodai-explain-v1",
    "fallback_used": false, "fallback_reason": null, "attempted_vendors": []
  }
}
```

Se incluyen también preguntas `confirm_portion` y `missing_value` cuando
corresponda. Se devuelven como máximo cinco
observaciones, setenta preguntas y cinco sugerencias. El máximo de preguntas
cubre todos los alimentos sin cantidad, valores faltantes y restricciones de
una petición válida, sin omitirlos por truncamiento. Una restricción solicita
verificar ingredientes/preparación; nunca garantiza ausencia de alérgenos ni
seguridad. No se deducen ingredientes ocultos. Las sugerencias proponen revisar
si se registró la opción del nutriente y su porción, cuando esa opción existe
en el plan aprobado; ante valores superiores piden revisar porciones y posibles
duplicados. No proponen nuevos objetivos ni cantidades.

Con `FOOD_AI_EXPLAIN_LLM_ENABLED=true`, la cadena Anthropic → OpenAI opcional
selecciona IDs de afirmaciones permitidas. El proveedor recibe esos IDs y sus
referencias de evidencia, los objetivos del plan aprobado, diferencias
calculadas, nombres y cantidades de alimentos confirmados, restricciones y
valores faltantes. Ese contexto permite priorizar observaciones relevantes.
Se excluyen identidad, ID de petición, fotos y memoria de conversación. Los
textos de entrada se tratan como datos, nunca como instrucciones. La respuesta
de proveedor se valida con esquema estricto y contra la lista permitida;
cualquier texto/campo extra o ID desconocido se rechaza. Las preguntas sobre
incertidumbre y restricciones se conservan siempre. Todos los textos publicados
son plantillas ES/EN del servidor. Si falla el proveedor, la validación o el
plazo compartido (3 segundos por defecto, configurable hasta 8), se usa la
respuesta determinista. El selector no reintenta; conserva circuitos separados
por proveedor. No se altera la cadena de visión Anthropic → OpenAI → legacy.

## Análisis y estado operativo

`POST /analyze` acepta un máximo de 10 MiB de archivo, 8192 píxeles por lado y
24 000 000 píxeles decodificados. Se revisan dimensiones antes de convertir o
inferir; los avisos/errores de bomba de descompresión se rechazan. Las imágenes
deben ser estáticas y tener firma/formato/MIME consistente. Se aplica orientación
EXIF antes de todas las inferencias y se eliminan metadatos del objeto de imagen.
Una imagen excesiva devuelve 400 `IMAGE_DIMENSIONS_EXCEEDED` sin inferencia.

La respuesta agrega `food_detected` (`false` con `foods: []` es un resultado
exitoso, sin activar fallback), `provenance` con proveedor/vendor/modelo,
`schema_version: foodai-analysis-v2`, `prompt_version: foodai-vision-v2` para LLM
y metadatos de fallback. La confianza visual usa `confidence_kind: model_score`:
es un score del modelo, **no una probabilidad calibrada**. El LLM agrega
`confidence_source: llm_self_report`. La porción indica
`estimate_kind: visual_estimate`, `requires_confirmation: true`; su confianza es
`model_score` para LLM y `heuristic_score` para estimadores legacy. Los gramos
son estimaciones, nunca un peso medido. La nutrición LLM por 100 g sigue siendo
una estimación y conserva su nombre explícito; USDA mantiene prioridad backend.

`/health` y `/health/readiness` agregan `capabilities`: proveedores configurados,
si hubo una petición válida en este proceso y capacidad de explicación
determinista. `configured: true` no acredita una clave válida.
`credentials_verified: false` / `verification: not_verified` permanece hasta
una respuesta válida; `successful_request_this_process` describe una observación
pasada, no una garantía de disponibilidad futura. Los health checks no hacen
requests externos ni pagos. Los estados legacy de carga se mantienen visibles.

## Configuración y despliegue manual

- `FOOD_AI_INTERNAL_SERVICE_KEY`: secreto compartido con
  `FoodAi:InternalServiceKey` del backend, nunca enviado al navegador. Vacío
  conserva compatibilidad de `/analyze`; la explicación responde 503
  `SERVICE_AUTH_NOT_CONFIGURED`. Configurado, ambos endpoints requieren la
  cabecera válida (401 si falta/es incorrecta).
- `FOOD_AI_EXPLAIN_LLM_ENABLED=false`: este es el valor por defecto. La
  explicación es determinista y declara `provider: deterministic`; las
  plantillas funcionan sin claves, llamadas pagas ni conexión al proveedor.
  Cambiarlo a `true` permite el selector LLM de evidencia mínima y exige claves
  servidor válidas. No se ha habilitado ni desplegado ese cambio de configuración.
- `FOOD_AI_EXPLAIN_TIMEOUT_SECONDS=3`: plazo total del selector, máximo 8.
- Las claves de visión existentes permanecen solo en entorno/gestor de secretos.

El operador debe crear el secreto real, conceder lectura únicamente a las tareas
backend y FoodAI que lo necesitan, configurar ambos servicios con el mismo
valor y permitir tráfico privado backend → FoodAI. Conservar health checks
privados y permitir salida a proveedores solo si se habilita LLM. No se incluye
ninguna credencial ni se ha realizado despliegue, push o validación pagada.

## Verificación local sin modelos ni proveedores reales

`e2e/service_app.py` usa el factory de producto `create_app(config=..., runtime=...)`
con un detector/estimador sintético y transporte que solo puede conectarse a un
proveedor HTTP guionado en localhost. El entry point de producción nunca importa
este módulo. No hay flag de entorno que sustituya modelos en producción.

```sh
env -u APPIMAGE -u APPDIR FOODAI_E2E_PORT=8012 /tmp/foodai-closure-venv/bin/python -m uvicorn e2e.service_app:app --host 127.0.0.1 --port 8012
FOODAI_SERVICE_URL=http://127.0.0.1:8012 node e2e/foodai-service.mjs all
FOODAI_SERVICE_URL=http://127.0.0.1:8012 node e2e/foodai-service.mjs providers
FOODAI_SERVICE_URL=http://127.0.0.1:8012 node e2e/foodai-service.mjs adverse
```

La suite usa Playwright API contra FastAPI real: autenticación, esquema/IDs,
diferencias inválidas, totales desconocidos, ES/EN, lenguaje humano para proteína
baja, opción/porción del plan, evidencia mínima del selector, EXIF/límites, no-food,
Anthropic → OpenAI → legacy, salida malformada, claims inventados, deadline y
health sin llamadas externas. Es evidencia suplementaria de contrato HTTP.
La evidencia E2E del producto requiere además la suite UI de la aplicación
paciente con API, Auth, Postgres y este servicio reales; no se presenta una UI
estática de prueba como el producto.

Antes de cambios de producto, las pruebas Playwright contra FastAPI real
fallaron por endpoint de explicación ausente (404), análisis sin autenticación
(200 en lugar de 401) y dimensión 8193×1 aceptada (200 en lugar de 400).
Después, las suites de contrato pasan. Las regresiones existentes enfocadas
pasan: 161 tests, tres skips y un caso deseleccionado; no se agregaron unit
tests. El caso deseleccionado (`test_retrieval_disabled_no_rompe`) importa
`torch` incluso en el camino desactivado y no se puede ejecutar en este entorno
ligero. La colección completa requiere además `torch` para tres módulos de
prueba excluidos (`test_hybrid_detector.py`, `test_zeroshot_classifier.py`,
`test_zeroshot_real.py`). La selección enfocada de análisis/LLM/porciones/
nutrición/esquemas pasó 111 tests sin deselecciones, con dos skips opcionales.
La mejora de redacción tuvo pruebas Playwright primero: fallaron el insight
humano de proteína baja y la falta de contexto aprobado en el selector; ambas
pasaron tras la implementación. Los escenarios locales `analyze_failure`
(503 de producto) y `analyze_timeout` (proveedor guionado demora tres segundos)
permiten probar recuperación HTTP sin proveedores reales. La reserva durable
y el XP se verifican en el E2E de backend/aplicación, no en este proceso Python.
No se descargaron pesos/datasets ni se usaron imágenes reales.
