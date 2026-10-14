import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateModel,validateInferredModel,validateLocalModel,inferredStep,localNeuralStep,plan,rollout,physicsStep,evaluatePath} from '../app/engine.mjs';
const read=p=>JSON.parse(fs.readFileSync(new URL(p,import.meta.url),'utf8'));
const inferred=validateInferredModel(read('../assets/inferred-model.json')),original=validateModel(read('../assets/model.json'));

test('browser inference of the new approach matches the Python wrapper, including wall and bumper steps',()=>{
  const f=read('./inferred-parity.json');
  assert.ok(f.states.length>=64);
  f.states.forEach((s,i)=>{const copy=s.slice(),p=inferredStep(s,inferred);assert.deepEqual(s,copy);p.forEach((v,j)=>assert.ok(Math.abs(v-f.nextStates[i][j])<2e-6,`state ${i} component ${j}`));});
});
test('the inferred geometry is recovered from the data and is close to the true table',()=>{
  const g=inferred.geometry,info=read('../assets/inferred-results.json');
  assert.ok(Math.abs(g.wall_lo-.02)<1e-3&&Math.abs(g.wall_hi-.98)<1e-3&&Math.abs(g.cx-.5)<1e-3&&Math.abs(g.cy-.5)<1e-3&&Math.abs(g.radius-.115)<1e-3);
  assert.equal(info.results.expanded.held_out.hits,96);assert.equal(info.results.original.held_out.hits,94);
});
test('invalid inferred geometry is rejected',()=>{
  assert.throws(()=>validateInferredModel({...inferred,geometry:{...inferred.geometry,radius:-1}}));
  assert.throws(()=>validateInferredModel({...inferred,geometry:{...inferred.geometry,wall_lo:.9,wall_hi:.1}}));
  assert.throws(()=>validateInferredModel({...inferred,geometry:undefined}));
});
test('geometry really matters: wrong inferred numbers change the prediction',()=>{
  const wrong={...inferred,geometry:{...inferred.geometry,radius:.25}},s=[.4,.5,.03,0];
  assert.notDeepEqual(inferredStep(s,wrong),inferredStep(s,inferred));
});
test('default geometry keeps the known-geometry model unchanged',()=>{
  const local=validateLocalModel(read('../assets/local-model.json'));
  const s=[.45,.5,.03,.0];
  assert.deepEqual(localNeuralStep(s,local),localNeuralStep(s,local,{wall_lo:.02,wall_hi:.98,cx:.5,cy:.5,radius:.115}));
});
test('planning with the new approach reaches the Across target in the real simulator',()=>{
  const start=[.16,.7],target=[.83,.72];
  const r=plan(original,{start,target,maxSpeed:.11,budget:512,steps:35,seed:123,controller:'inferred-neural',inferredModel:inferred});
  const actual=evaluatePath(rollout(start,r.action,35,physicsStep),target);
  assert.ok(actual.hit,`actual miss ${actual.error}`);
});
