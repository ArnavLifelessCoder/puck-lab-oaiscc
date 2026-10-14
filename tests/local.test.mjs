import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateModel,validateLocalModel,localNeuralStep,validateTeamModel,teamStep,plan,rollout,physicsStep,evaluatePath} from '../app/engine.mjs';
const read=p=>JSON.parse(fs.readFileSync(new URL(p,import.meta.url),'utf8'));
const local=validateLocalModel(read('../assets/local-model.json')),original=validateModel(read('../assets/model.json'));
test('local-coordinate neural inference matches independent Python/PyTorch values',()=>{
 const f=read('../assets/local-parity.json');
 f.states.forEach((s,i)=>localNeuralStep(s,local).forEach((v,j)=>assert.ok(Math.abs(v-f.nextStates[i][j])<2e-6)));
});
test('zero learned responses never secretly invoke physics, including walls and bumper',()=>{
 const zero=structuredClone(local);
 for(const k of ['freeWeight','wallWeight','skipWeight'])zero[k].fill(0);
 for(const k of ['weights','biases'])zero[k]=zero[k].map(a=>a.map(()=>0));
 for(const s of [[.2,.2,.02,0],[.97,.97,.03,.03],[.38,.5,.02,0],[.5,.4,0,.1]])assert.deepEqual(localNeuralStep(s,zero),s);
});
test('new browser planner reproduces all 24 development selections and scores',()=>{
 const report=read('../host/results/local-neural-development.json');let hits=0;
 for(const row of report.rows.filter(r=>r.controller==='improved-neural')){
  const scene=report.scenes[row.scene],shot=plan(original,{...scene,steps:35,budget:2048,maxSpeed:.11,controller:'improved-neural',improvedModel:local});
  shot.action.forEach((v,i)=>assert.ok(Math.abs(v-row.action[i])<1e-12));
  const actual=evaluatePath(rollout(scene.start,shot.action,35,physicsStep),scene.target);
  assert.ok(Math.abs(actual.error-row.actual_miss)<1e-9);assert.equal(actual.hit,row.hit);hits+=actual.hit;
 }
 assert.equal(hits,24);
});
test('student checkpoints with both raw and geometry inputs match Python in browser',()=>{
 for(const f of read('./team-parity.json')){
  const m=validateTeamModel(f.model);
  f.states.forEach((s,i)=>teamStep(s,m).forEach((v,j)=>assert.ok(Math.abs(v-f.nextStates[i][j])<1e-10)));
  const bad=structuredClone(m);bad.inputScale[0]=0;assert.throws(()=>validateTeamModel(bad));
  const huge=structuredClone(m);huge.architecture[1]=100000;assert.throws(()=>validateTeamModel(huge));
 }
});
