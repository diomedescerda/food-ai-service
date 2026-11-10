/** Supplementary Playwright API contract checks against real FastAPI HTTP.
 * The product UI integration is owned by the patient app's FoodAI E2E suite.
 * No page.setContent/static fake UI is used or presented as product E2E.
 * Start: FOOD_AI_INTERNAL_SERVICE_KEY=foodai-e2e-secret python -m uvicorn e2e.service_app:app --port 18980
 * Run: node e2e/foodai-service.mjs [explain|upload|all]
 */
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
const require = createRequire(new URL('../../coppaddresd-front/package.json', import.meta.url));
const { request } = require('playwright');
const api = await request.newContext({ baseURL: process.env.FOODAI_SERVICE_URL || 'http://127.0.0.1:18980' });
const key = process.env.FOODAI_TEST_SERVICE_KEY || 'foodai-e2e-secret';
const requestId = '3f3f0f0f-1111-2222-3333-444444444444';
export const explainPayload = {
  schema_version: '1', request_id: requestId, locale: 'es',
  plan: { id: 'plan-1', version: 'approved-1', targets: [{ id: 'target-calories', metric: 'calories', amount: 2000, unit: 'kcal' }] },
  diffs: [{ id: 'diff-calories', target_id: 'target-calories', metric: 'calories', actual: 450, target: 2000, delta: -1550, unit: 'kcal' }],
  confirmed_foods: [{ id: 'food-rice', name: 'Arroz', grams: null }],
  restrictions: [{ id: 'restriction-1', label: 'Maní' }],
  missing_values: [{ id: 'missing-protein', field: 'protein_g', label: 'Proteína sin verificar' }],
};
const postExplain = (data, secret = key) => api.post('/internal/foodai/explain', { headers: { 'X-FoodAI-Service-Key': secret }, data });
async function explanationChecks() {
  assert.equal((await postExplain(explainPayload, 'wrong-key')).status(), 401, 'internal explanation requires service authentication');
  const response = await postExplain(explainPayload);
  assert.equal(response.status(), 200, await response.text());
  const body = await response.json();
  assert.equal(body.schema_version, '1');
  assert.equal(body.locale, 'es');
  assert.ok(body.observations.some(item => item.claim === 'metric_diff' && item.evidence_ids.includes('diff-calories')));
  assert.ok(body.questions.some(item => item.claim === 'confirm_portion' && item.evidence_ids.includes('food-rice')));
  assert.ok(body.questions.some(item => item.claim === 'missing_value' && item.evidence_ids.includes('missing-protein')));
  const ids = new Set(['plan-1','target-calories','diff-calories','food-rice','restriction-1','missing-protein']);
  for (const item of [...body.observations,...body.questions,...body.suggestions]) {
    assert.ok(item.evidence_ids.length && item.evidence_ids.every(id => ids.has(id)));
    assert.ok(!/garantiza|sin alérgenos|allergen.free|safe to eat/i.test(item.text));
  }
  assert.equal(body.provenance.prompt_version, 'foodai-explain-v1');
  assert.equal(body.provenance.vendor, 'local');
  for (const bad of [
    { ...explainPayload, invented: 999 },
    { ...explainPayload, diffs: [{ ...explainPayload.diffs[0], delta: 100 }] },
    { ...explainPayload, diffs: [{ ...explainPayload.diffs[0], target_id: 'unknown' }] },
  ]) assert.equal((await postExplain(bad)).status(), 422, 'strict schema rejects ungrounded facts');
  const partial = structuredClone(explainPayload);
  partial.locale = 'en'; partial.diffs[0].actual = null; partial.diffs[0].delta = null;
  const english = await (await postExplain(partial)).json();
  assert.equal(english.locale, 'en');
  assert.deepEqual(english.observations, [], 'unknown totals must never yield a numeric comparison');
  assert.ok(english.questions.some(item => /confirm|missing|verify/i.test(item.text)));
  console.log('PASS explanation: auth, strict evidence, invalid deltas, missing data, ES/EN');
}
function syntheticImage(width, height, exif = false) {
  const python = process.env.FOODAI_PYTHON || '/tmp/foodai-closure-venv/bin/python';
  const env = { ...process.env }; delete env.APPIMAGE; delete env.APPDIR;
  return execFileSync(python, ['-c', `from PIL import Image; from io import BytesIO; import sys; im=Image.new('RGB',(${width},${height}),(200,120,30)); b=BytesIO(); ex=Image.Exif(); ex[274]=6; im.save(b,format='JPEG',**({'exif':ex} if ${exif ? 'True' : 'False'} else {})); sys.stdout.buffer.write(b.getvalue())`], { env });
}
async function uploadChecks() {
  const send = (buffer) => api.post('/analyze', { headers: { 'X-FoodAI-Service-Key': key }, multipart: { analysis_id: requestId, image: { name: 'synthetic.jpg', mimeType: 'image/jpeg', buffer } } });
  assert.equal((await api.post('/analyze', {multipart:{ analysis_id:requestId,image:{name:'synthetic.jpg',mimeType:'image/jpeg',buffer:syntheticImage(8,8)}}})).status(), 401);
  const beforeLimits = (await (await api.get('/__e2e/state')).json()).detector_calls;
  for (const [w,h] of [[8193,1],[5000,5000]]) {
    const response = await send(syntheticImage(w,h));
    assert.equal(response.status(),400,`decoded image ${w}x${h} rejected before inference`);
    assert.equal((await response.json()).detail.error.code,'IMAGE_DIMENSIONS_EXCEEDED');
  }
  assert.equal((await (await api.get('/__e2e/state')).json()).detector_calls,beforeLimits,'decoded limits reject before inference');
  const response = await send(syntheticImage(8,12,true));
  assert.equal(response.status(),200,await response.text());
  const body = await response.json();
  assert.equal(body.provenance.schema_version,'foodai-analysis-v2');
  assert.equal(body.provenance.vendor,'local');
  assert.equal(body.foods[0].confidence_kind,'model_score');
  assert.equal(body.foods[0].portion.estimate_kind,'visual_estimate');
  const state = await (await api.get('/__e2e/state')).json();
  assert.deepEqual(state.image_size,[12,8], 'EXIF orientation normalized before detector');
  assert.deepEqual(state.image_info_keys,[], 'image metadata stripped before inference');
  console.log('PASS upload: service auth, decoded dimensions/pixels, provenance');
}
async function providerChecks() {
  const set = async name => assert.equal((await api.post('/__e2e/scenario',{data:{name}})).status(),200);
  const state = async () => (await api.get('/__e2e/state')).json();
  const send = () => api.post('/analyze',{headers:{'X-FoodAI-Service-Key':key},multipart:{analysis_id:requestId,image:{name:'synthetic.jpg',mimeType:'image/jpeg',buffer:syntheticImage(16,16)}}});
  for (const [scenario,vendor,fallback] of [['anthropic','anthropic',false],['backup','openai',true],['both_fail','local',true],['malformed_vision','local',true]]) {
    await set(scenario);
    const readiness = await (await api.get('/health/readiness')).json();
    assert.equal(readiness.capabilities.providers.anthropic.configured,true);
    assert.equal(readiness.capabilities.providers.anthropic.credentials_verified,false);
    assert.equal(readiness.capabilities.health_uses_external_probes,false);
    await api.get('/health');
    assert.equal((await state()).provider_requests.length,0,'health/readiness never call a paid provider');
    const response = await send();
    assert.equal(response.status(),200,await response.text());
    const body = await response.json();
    assert.equal(body.provenance.vendor,vendor);
    assert.equal(body.provenance.fallback_used,fallback);
    assert.equal(body.food_detected,true);
    assert.equal(body.foods[0].portion.requires_confirmation,true);
    assert.ok(!JSON.stringify(body).includes(key));
    if (vendor !== 'local') {
      const after = await (await api.get('/health/readiness')).json();
      assert.equal(after.capabilities.providers[vendor].credentials_verified,true);
      assert.equal(after.capabilities.providers[vendor].verification,'successful_request_this_process');
    }
  }
  await set('no_food');
  const before = (await state()).detector_calls;
  const empty = await (await send()).json();
  assert.deepEqual(empty.foods,[]);
  assert.equal(empty.food_detected,false);
  assert.equal(empty.status,'completed');
  assert.equal(empty.provenance.vendor,'anthropic');
  assert.equal((await state()).detector_calls,before,'explicit no-food success does not invoke fallback');
  for (const [scenario,vendor,fallback] of [['explain_selection','anthropic',false],['explain_backup','openai',true],['explain_openai','openai',false],['invalid_claim','local',true],['unknown_claim','local',true],['explain_timeout','local',true]]) {
    await set(scenario);
    const start = Date.now();
    const response = await postExplain(explainPayload);
    assert.equal(response.status(),200,await response.text());
    const body = await response.json();
    assert.equal(body.provenance.vendor,vendor);
    assert.equal(body.provenance.fallback_used,fallback);
    assert.ok(body.observations.every(item => item.evidence_ids.includes('diff-calories')));
    assert.ok(!JSON.stringify(body).includes('999'));
    if(scenario==='explain_timeout') assert.ok(Date.now()-start<1200,'single bounded explanation deadline, deterministic fallback');
    const sent = JSON.stringify((await state()).provider_requests);
    assert.ok(!sent.includes('image_url')&&!sent.includes('base64')&&!sent.includes(requestId),'selector excludes images and request identity');
  }
  await set('legacy');
  console.log('PASS provider chain: Anthropic → OpenAI → legacy, no-food, invalid schema, explanation claims/deadline, no-cost health');
}
const lowProteinPayload = {
  ...explainPayload,
  plan: {id:'plan-1',version:'approved-1',targets:[{id:'target-protein',metric:'protein_g',amount:100,unit:'g'}]},
  diffs: [{id:'diff-protein',target_id:'target-protein',metric:'protein_g',actual:20,target:100,delta:-80,unit:'g'}],
  confirmed_foods:[{id:'food-lentils',name:'Arroz con lentejas',grams:200}],
  missing_values:[],
};
async function humanCopyChecks() {
  for(const locale of ['es','en']) {
    const response=await postExplain({...lowProteinPayload,locale});
    assert.equal(response.status(),200,await response.text());
    const body=await response.json();
    const observation=body.observations.find(item=>item.claim==='metric_diff');
    assert.ok(observation.text.startsWith(locale==='es'?'Tus alimentos confirmados aportan 20 g de proteína':'Your confirmed foods provide 20 g of protein'),'human low-protein observation describes confirmed evidence');
    assert.ok(observation.text.includes('100 g')&&observation.text.includes('80 g'),'only approved target and calculated difference are used');
    assert.ok(!/registro confirmado|diferencia|confirmed record|difference/i.test(observation.text),'copy avoids technical calculation labels');
    const suggestion=body.suggestions.find(item=>item.claim==='review_plan');
    assert.ok(locale==='es'?/opción de proteína/.test(suggestion.text)&&/porción/.test(suggestion.text):/protein option/.test(suggestion.text)&&/portion/.test(suggestion.text),'useful next step checks approved protein option and portion');
    assert.ok(suggestion.evidence_ids.includes('diff-protein'));
    assert.ok(!/\d/.test(suggestion.text),'suggestion invents no new quantities');
    assert.equal(body.provenance.provider,'deterministic');
  }
  console.log('PASS human copy: ES/EN low-protein insight and grounded approved-option/portion next step');
}
async function selectorEvidenceChecks() {
  await api.post('/__e2e/scenario',{data:{name:'explain_selection'}});
  const response=await postExplain(lowProteinPayload);
  assert.equal(response.status(),200,await response.text());
  const body=await response.json();
  assert.equal(body.provenance.provider,'llm');
  const state=await(await api.get('/__e2e/state')).json();
  const content=state.provider_requests[0].payload.messages[0].content;
  const approved=JSON.parse(content.split('\n').slice(1).join('\n'));
  assert.ok(approved.evidence,'selector is given approved evidence to make a relevant selection');
  assert.deepEqual(approved.evidence.plan,lowProteinPayload.plan,'selector receives approved targets');
  assert.deepEqual(approved.evidence.diffs,lowProteinPayload.diffs,'selector receives calculated differences');
  assert.deepEqual(approved.evidence.confirmed_foods,lowProteinPayload.confirmed_foods,'selector receives confirmed food names and amounts');
  assert.deepEqual(approved.evidence.restrictions,lowProteinPayload.restrictions);
  assert.deepEqual(approved.evidence.missing_values,[]);
  assert.ok(approved.allowed_claims.every(claim=>claim.evidence_ids.length));
  assert.deepEqual(Object.keys(approved.evidence).sort(),['confirmed_foods','diffs','missing_values','plan','restrictions']);
  assert.ok(!content.includes(requestId)&&!content.includes('image_url')&&!content.includes('base64'),'no identity/photo/chat memory in selector context');
  await api.post('/__e2e/scenario',{data:{name:'legacy'}});
  console.log('PASS selector context: minimal approved evidence, ID-only claims, no identity/photo/chat');
}
async function adverseAnalyzeChecks() {
  const set=async name=>api.post('/__e2e/scenario',{data:{name}});
  const send=(timeout=10_000)=>api.post('/analyze',{timeout,headers:{'X-FoodAI-Service-Key':key},multipart:{analysis_id:requestId,image:{name:'synthetic.jpg',mimeType:'image/jpeg',buffer:syntheticImage(16,16)}}});
  await set('analyze_failure');
  const failed=await send();
  assert.equal(failed.status(),503);
  assert.equal((await failed.json()).detail.error.code,'MODEL_NOT_READY');
  await set('legacy');
  assert.equal((await send()).status(),200,'same analysis ID can succeed after external failure');
  await set('analyze_timeout');
  await assert.rejects(()=>send(100),/Timeout 100ms/,'local scripted provider can exercise real HTTP timeout');
  await set('legacy');
  assert.equal((await send()).status(),200,'same analysis ID can succeed after caller timeout');
  console.log('PASS adverse analysis boundaries: explicit503, caller timeout, same-ID retry; durable reservation proof remains backend UI-owned');
}
try {
  const suite = process.argv[2] || 'all';
  if (suite === 'all' || suite === 'explain') await explanationChecks();
  if (suite === 'all' || suite === 'upload') await uploadChecks();
  if (suite === 'providers') await providerChecks();
  if (suite === 'all' || suite === 'human-copy') await humanCopyChecks();
  if (suite === 'all' || suite === 'selector-evidence') await selectorEvidenceChecks();
  if (suite === 'adverse') await adverseAnalyzeChecks();
} finally { await api.dispose(); }
