import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import * as engine from '../app/engine.mjs';

// Exercise the real app's execution handler with a deterministic browser frame scheduler.
function harness(){
 const widgets=new Map();let callback=null,drawFailure=false;
 const context2d=new Proxy({}, {get:(o,k)=>k==='clearRect'?()=>{if(drawFailure)throw Error('Test rendering failure');}:o[k]??(()=>{}),set:(o,k,v)=>(o[k]=v,true)});
 const widget=id=>{if(!widgets.has(id))widgets.set(id,{value:'512',disabled:false,hidden:false,textContent:'',listeners:{},addEventListener(name,fn){this.listeners[name]=fn;},getContext(){return context2d;},classList:{toggle(){}}});return widgets.get(id);};
 const sandbox={...engine,document:{getElementById:widget,querySelectorAll:()=>[]},performance:{now:()=>100},requestAnimationFrame:fn=>(callback=fn,1),cancelAnimationFrame(){callback=null;}};
 vm.createContext(sandbox);
 const source=fs.readFileSync(new URL('../app/app.mjs',import.meta.url),'utf8').replace(/^import .*;\r?\n/,'').replace(/load\(\);\s*$/,'');
 vm.runInContext(source,sandbox);
 vm.runInContext("planned={action:[.025,0],path:rollout(start,[.025,0],steps,physicsStep)};planned.assessment=evaluatePath(planned.path,target);currentConfig={controller:'calibrated',budget:512,maxSpeed:.11};",sandbox);
 return {widget,run:code=>vm.runInContext(code,sandbox),tick:time=>{assert.ok(callback,'a frame must be scheduled');const fn=callback;callback=null;fn(time);},setDrawFailure:()=>{drawFailure=true;},hasFrame:()=>callback!==null};
}
test('earlier first-frame timestamp cannot index a negative trajectory frame or lock controls',()=>{
 const h=harness();h.run('execute()');h.tick(99);
 assert.equal(h.run('frame'),0);assert.equal(h.run('busy'),true);assert.equal(h.widget('error').hidden,true);
 h.tick(98);assert.equal(h.run('frame'),0);
 h.tick(750);assert.equal(h.run('frame'),10);
 h.tick(2500);assert.equal(h.run('frame'),35);assert.equal(h.run('busy'),false);
 assert.equal(h.run('history.length'),1);assert.equal(h.widget('plan').disabled,false);
 assert.equal(h.widget('reset').disabled,false);assert.equal(h.hasFrame(),false);
 assert.equal(h.widget('stage').textContent,'03 / EXPLAIN THE DIFFERENCE');
});
test('rendering failure is reported and unlocks controls without recording a successful run',()=>{
 const h=harness();h.run('execute()');h.setDrawFailure();h.tick(99);
 assert.equal(h.run('busy'),false);assert.equal(h.run('history.length'),0);
 assert.equal(h.widget('error').hidden,false);assert.match(h.widget('error').textContent,/Test rendering failure/);
 assert.equal(h.widget('reset').disabled,false);assert.equal(h.widget('plan').disabled,false);
 assert.equal(h.hasFrame(),false);
});
test('a delayed frame completes a shot once instead of leaving it executing',()=>{
 const h=harness();h.run('execute()');h.tick(100);h.tick(60000);
 assert.equal(h.run('frame'),35);assert.equal(h.run('history.length'),1);
 assert.equal(h.run('busy'),false);assert.equal(h.hasFrame(),false);
});

test('team upload selects valid neural weights and rejects bad files without locking controls',async()=>{
 const h=harness(),fixture=JSON.parse(fs.readFileSync(new URL('./team-parity.json',import.meta.url),'utf8'))[0];
 h.run('worker={postMessage(){}}');
 const input=h.widget('team-file');
 await input.listeners.change({target:{files:[{name:'network.json',size:1000,text:async()=>JSON.stringify(fixture.model)}],value:'network.json'}});
 assert.equal(h.widget('controller').value,'team');assert.equal(h.widget('team-option').disabled,false);
 assert.equal(h.run('busy'),false);assert.equal(h.widget('error').hidden,true);
 await input.listeners.change({target:{files:[{name:'bad.json',size:1000,text:async()=>'{broken'}],value:'bad.json'}});
 assert.equal(h.widget('error').hidden,false);assert.equal(h.run('busy'),false);assert.equal(h.widget('plan').disabled,false);
 assert.equal(h.widget('controller').value,'team');
});

test('switching models can predict and execute the identical strong push',()=>{
 const h=harness(),read=n=>JSON.parse(fs.readFileSync(new URL(`../assets/${n}.json`,import.meta.url),'utf8'));
 h.run(`model=${JSON.stringify(read('model'))};improvedModel=${JSON.stringify(read('local-model'))};parameters=${JSON.stringify(read('calibration').parameters)};`);
 h.widget('speed-range').value='.11';h.widget('controller').value='improved-neural';
 h.run('rememberedPush=[-.03231376739071546,-.10113657323820864];samePush();');
 const selected=h.run('JSON.stringify(planned.action)');
 h.run('execute()');h.tick(0);h.tick(3000);const actual=h.run('JSON.stringify(actual)');
 h.widget('controller').value='neural';h.widget('controller').listeners.change();
 assert.equal(h.widget('same-push').disabled,false);
 h.run('samePush()');assert.equal(h.run('JSON.stringify(planned.action)'),selected);
 assert.equal(h.run('currentConfig.samePush'),true);
 h.run('execute()');h.tick(0);h.tick(3000);
 assert.equal(h.run('JSON.stringify(actual)'),actual);
 assert.match(h.widget('route-info').textContent,/limit 0.110; 35 steps/);
 assert.equal(h.run('history[0].search'),'Same push');
});
test('manual shots and remembered shots obey the same reduced push limit',()=>{
 const h=harness();h.widget('speed-range').value='.04';h.widget('power').value='110';
 h.run('rememberedPush=[.10,0]');h.widget('speed-range').listeners.change();
 assert.equal(Number(h.widget('power').value),40);assert.equal(Number(h.widget('power').max),40);
 assert.equal(h.widget('same-push').disabled,true);
 h.run('samePush()');assert.equal(h.run('planned'),null);
 for(const controller of ['neural','improved-neural','calibrated']){
  h.widget('controller').value=controller;h.widget('angle').value='0';h.widget('power').value='110';
  h.run('manualShot()');assert.equal(h.run('planned'),null);assert.match(h.widget('status').textContent,/exceeds the selected limit/);
 }
});
