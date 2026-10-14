import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {validateModel,learnedStep,physicsStep,rollout,plan,evaluatePath} from '../app/engine.mjs';
const read=n=>JSON.parse(fs.readFileSync(new URL(`../assets/${n}.json`,import.meta.url),'utf8'));
const model=validateModel(read('model')),parity=read('parity');
test('JavaScript inference and simulator match independent Python calculations',()=>{
  parity.states.forEach((s,i)=>{
    learnedStep(s,model).forEach((v,j)=>assert.ok(Math.abs(v-parity.nextStates[i][j])<1e-10));
    physicsStep(s).forEach((v,j)=>assert.ok(Math.abs(v-parity.physicsNextStates[i][j])<1e-10));
  });
});
test('wall and bumper contact reflect motion, dissipate speed, and do not mutate input',()=>{
  const s=[.97,.2,.03,0],copy=s.slice(),next=physicsStep(s);
  assert.ok(next[0]<.98&&next[2]<0);assert.deepEqual(s,copy);
  const bump=physicsStep([.37,.5,.02,0]);assert.ok(bump[2]<0);assert.ok(Math.abs(bump[0]-.385)<1e-10);
});
test('planner is reproducible and larger search includes all smaller-search candidates',()=>{
  const config={start:[.16,.7],target:[.83,.72],steps:35,budget:32,maxSpeed:.11};
  const a=plan(model,config),b=plan(model,{...config,budget:128});
  assert.deepEqual(a,plan(model,config));assert.ok(b.cost<=a.cost);
  assert.ok(a.speed>=.008&&a.speed<=.11);
  assert.deepEqual(a.path,rollout(config.start,a.action,35,s=>learnedStep(s,model)));
});
test('zero motion stays still away from contact and target scoring uses actual endpoint',()=>{
  const path=rollout([.2,.2],[0,0],35,physicsStep);
  assert.ok(path.every(s=>s[0]===.2&&s[1]===.2));
  assert.equal(evaluatePath(path,[.2,.2]).hit,true);
  assert.equal(evaluatePath(path,[.8,.8]).hit,false);
});

test('calibrated and contact-repaired JS transitions match independent Python fixtures',async()=>{
 const {structuredStep,validateCalibration}=await import('../app/engine.mjs');
 const fixture=read('controller-parity'),parameters=validateCalibration(read('calibration').parameters);
 fixture.states.forEach((s,i)=>{
  structuredStep(s,parameters).forEach((v,j)=>assert.ok(Math.abs(v-fixture.calibrated[i][j])<1e-10));
  structuredStep(s,parameters,model).forEach((v,j)=>assert.ok(Math.abs(v-fixture.contactRepair[i][j])<1e-10));
 });
 assert.throws(()=>validateCalibration({drag:NaN,wall_restitution:.8,bumper_restitution:.8}));
});
test('reference plans via fitted coefficients and succeeds without changing action budget',()=>{
 const parameters=read('calibration').parameters,config={start:[.16,.7],target:[.83,.72],steps:35,budget:2048,maxSpeed:.11,controller:'calibrated',parameters};
 const reference=plan(model,config),actual=rollout(config.start,reference.action,35,physicsStep);
 assert.ok(evaluatePath(actual,config.target).hit);
 reference.path.forEach((s,i)=>s.forEach((v,j)=>assert.ok(Math.abs(v-actual[i][j])<1e-9)));
 assert.equal(reference.path.length,36);
});
