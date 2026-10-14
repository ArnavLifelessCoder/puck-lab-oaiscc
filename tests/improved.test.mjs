import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateModel,validateImprovedModel,improvedStep,physicsStep,rollout,plan,evaluatePath} from '../app/engine.mjs';
const read=p=>JSON.parse(fs.readFileSync(new URL(p,import.meta.url),'utf8'));
const original=validateModel(read('../assets/model.json'));
const improved=validateImprovedModel(read('../assets/improved-model.json'));
const fixture=read('../assets/improved-parity.json');
test('browser neural branches match independent PyTorch validation predictions',()=>{
 let maximum=0;
 for(let i=0;i<fixture.states.length;i++){
  const state=fixture.states[i],copy=state.slice(),prediction=improvedStep(state,improved);
  for(let j=0;j<4;j++)maximum=Math.max(maximum,Math.abs(prediction[j]-fixture.nextStates[i][j]));
  assert.deepEqual(state,copy);
 }
 assert.ok(maximum<2e-6,`PyTorch/browser difference ${maximum}`);
});
test('learned weights determine contact and free responses; no physics override',()=>{
 const zero=structuredClone(improved);
 for(const k of ['weights','biases'])zero[k]=zero[k].map(a=>a.map(()=>0));
 for(const k of ['skipWeight','skipBias','freeWeight','freeBias'])zero[k].fill(0);
 for(const state of [[.2,.2,.01,0],[.97,.2,.03,0],[.38,.5,.02,0]])assert.deepEqual(improvedStep(state,zero),state);
 const invalid=structuredClone(improved);invalid.freeWeight[0]=NaN;
 assert.throws(()=>validateImprovedModel(invalid));
 const short=structuredClone(improved);short.weights[1].pop();assert.throws(()=>validateImprovedModel(short));
});
test('browser planner reproduces all 24 Python development selections and outcomes',()=>{
 const report=read('../host/results/neural-development.json');let hits=0;
 for(const row of report.rows.filter(r=>r.controller==='improved-neural')){
  const scene=report.scenes[row.scene];
  const shot=plan(original,{...scene,steps:35,budget:2048,maxSpeed:.11,seed:123,controller:'improved-neural',improvedModel:improved});
  shot.action.forEach((v,j)=>assert.ok(Math.abs(v-row.action[j])<1e-12));
  const actual=evaluatePath(rollout(scene.start,shot.action,35,physicsStep),scene.target);
  assert.equal(actual.hit,row.hit);assert.ok(Math.abs(actual.error-row.actual_miss)<1e-9);hits+=actual.hit;
 }
 assert.equal(hits,16);
});
