import fs from 'node:fs';
import {validateModel,plan,rollout,physicsStep,evaluatePath} from '../app/engine.mjs';
const root=new URL('../assets/',import.meta.url),model=validateModel(JSON.parse(fs.readFileSync(new URL('model.json',root),'utf8')));
const scenarios=[{start:[.16,.7],target:[.83,.72]},{start:[.16,.7],target:[.82,.25]},{start:[.16,.7],target:[.3,.24]},{start:[.8,.2],target:[.2,.75]},{start:[.18,.2],target:[.8,.3]},{start:[.7,.8],target:[.25,.8]}];
const rows=[];
for(const [scene,config] of scenarios.entries())for(const maxSpeed of [.04,.11])for(const budget of [32,128,512,2048]){
  const result=plan(model,{...config,maxSpeed,budget,steps:35,seed:123});
  const actual=evaluatePath(rollout(config.start,result.action,35,physicsStep),config.target);
  rows.push({scene,...config,maxSpeed,budget,predicted:result.assessment,actual,action:result.action,speed:result.speed});
}
const report={scope:'Exploratory six-scene sweep, including three scenes inspected during development. Not a held-out control benchmark or evidence of a universal search effect. Model fixed before examining scenes. Same candidate seed and horizon throughout.',rows};
fs.writeFileSync(new URL('../host/results/exploratory-scenes.json',import.meta.url),JSON.stringify(report,null,2));
for(const maxSpeed of [.04,.11])for(const budget of [32,128,512,2048]){const r=rows.filter(v=>v.maxSpeed===maxSpeed&&v.budget===budget);console.log({maxSpeed,budget,hits:r.filter(v=>v.actual.hit).length,total:r.length,meanPredictedMiss:r.reduce((s,v)=>s+v.predicted.error,0)/r.length,meanActualMiss:r.reduce((s,v)=>s+v.actual.error,0)/r.length});}
