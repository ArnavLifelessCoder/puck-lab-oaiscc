import {validateModel,validateImprovedModel,validateInferredModel,validateTeamModel,learnedStep,controllerStep,validateCalibration,physicsStep,rollout,evaluatePath,routeLength,distance,OBSTACLE,TARGET_RADIUS} from './engine.mjs';
const $=id=>document.getElementById(id),canvas=$('table'),ctx=canvas.getContext('2d');
const budgets=[32,128,512,2048],start=[.16,.7],steps=35;
const hostMode=document.documentElement?.dataset?.mode==='host';
let model,improvedModel,inferredModel,teamModel=null,parameters,worker,target=[.83,.72],planned=null,actual=null,frame=0,animation=0,busy=false,history=[],currentConfig=null,rememberedPush=null;
const controllerNames={'neural':'Original NN','improved-neural':'Known-geometry model','inferred-neural':'New approach (inferred geometry)','previous-neural':'Previous improved NN','team':'Your team NN','contact-repair':'Contact repair','calibrated':'Calibrated reference'};
const unit=n=>`${(n*100).toFixed(1)}%`,point=s=>[140+s[0]*720,45+s[1]*720];
function circle(x,y,r,fill,stroke,line=1){ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.lineWidth=line;ctx.strokeStyle=stroke;ctx.stroke();}}
function radial(x,y,r0,r1,stops,fallback){const g=ctx.createRadialGradient?.(x,y,r0,x,y,r1);if(!g||!g.addColorStop)return fallback;for(const [offset,color] of stops)g.addColorStop(offset,color);return g;}
function path(points,color,dashed=false,glow=false){ctx.save();ctx.strokeStyle=color;ctx.lineWidth=dashed?3:4;ctx.lineJoin='round';ctx.lineCap='round';ctx.setLineDash(dashed?[10,9]:[]);if(glow){ctx.shadowColor=color+'66';ctx.shadowBlur=10;}ctx.beginPath();points.forEach((s,i)=>{const p=point(s);i?ctx.lineTo(...p):ctx.moveTo(...p);});ctx.stroke();ctx.restore();}
function puck(p,color){ctx.save();ctx.shadowColor='rgba(15,23,42,.35)';ctx.shadowBlur=12;ctx.shadowOffsetY=4;circle(...p,13,color);ctx.restore();circle(...p,13,null,'#ffffff',3);circle(...p,4,'rgba(255,255,255,.85)');}
function draw(){
  ctx.clearRect(0,0,1000,820);
  // table surface
  ctx.save();ctx.shadowColor='rgba(30,58,138,.18)';ctx.shadowBlur=26;ctx.shadowOffsetY=8;ctx.fillStyle='#ffffff';ctx.fillRect(140,45,720,720);ctx.restore();
  ctx.fillStyle=radial(500,405,60,560,[[0,'#ffffff'],[1,'#eaf0fb']],'#f6f9ff');ctx.fillRect(140,45,720,720);
  ctx.strokeStyle='#e3eaf6';ctx.lineWidth=1;
  for(let i=0;i<=10;i++){const n=i*.1;ctx.beginPath();ctx.moveTo(...point([n,0]));ctx.lineTo(...point([n,1]));ctx.moveTo(...point([0,n]));ctx.lineTo(...point([1,n]));ctx.stroke();}
  // walls act at 0.02 and 0.98 of the table, so shade that strip
  const wall=.02*720;ctx.fillStyle='rgba(100,116,139,.12)';ctx.fillRect(140,45,720,wall);ctx.fillRect(140,765-wall,720,wall);ctx.fillRect(140,45,wall,720);ctx.fillRect(860-wall,45,wall,720);
  ctx.strokeStyle='#94a3b8';ctx.lineWidth=3;ctx.strokeRect(140,45,720,720);
  // bumper
  const ob=point([.5,.5]),bumper=OBSTACLE.r*720;
  ctx.save();ctx.shadowColor='rgba(15,23,42,.25)';ctx.shadowBlur=18;ctx.shadowOffsetY=6;circle(...ob,bumper,'#dbe3f1');ctx.restore();
  circle(...ob,bumper,radial(ob[0]-bumper*.3,ob[1]-bumper*.35,bumper*.1,bumper,[[0,'#f8fafc'],[1,'#c9d4e8']],'#dbe3f1'),'#8a9bb8',2);
  circle(...ob,bumper*.55,null,'#b3c0d9',1.5);
  ctx.fillStyle='#475569';ctx.font='700 13px Segoe UI';ctx.textAlign='center';ctx.fillText('BUMPER',ob[0],ob[1]+5);
  // target
  const t=point(target),ring=TARGET_RADIUS*720;
  circle(...t,ring,'rgba(22,163,74,.10)','#16a34a',2.5);circle(...t,ring*.62,null,'rgba(22,163,74,.55)',1.5);circle(...t,ring*.25,'#16a34a');
  ctx.fillStyle='#15803d';ctx.font='700 13px Segoe UI';ctx.fillText('TARGET',t[0],t[1]-ring-12);
  // predicted path (dashed violet) and actual path (solid blue)
  if(planned){path(planned.path,'#7c3aed',true);const end=point(planned.path.at(-1));circle(...end,10,'rgba(124,58,237,.14)','#7c3aed',2.5);}
  if(actual){
    path(actual.slice(0,frame+1),'#2563eb',false,true);
    if(planned&&frame>=actual.length-1){const a=point(planned.path.at(-1)),b=point(actual.at(-1));ctx.save();ctx.strokeStyle='#d97706';ctx.lineWidth=2.5;ctx.setLineDash([3,6]);ctx.lineCap='round';ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.restore();}
    puck(point(actual[Math.min(frame,actual.length-1)]),'#2563eb');
  }else{const p=point(start);puck(p,'#2563eb');ctx.fillStyle='#475569';ctx.font='600 13px Segoe UI';ctx.fillText('START',p[0],p[1]+32);}
  ctx.textAlign='left';ctx.fillStyle='#94a3b8';ctx.font='12px Consolas';ctx.fillText('0',120,47);ctx.fillText('1',870,770);
}
function setBusy(value){busy=value;for(const id of ['plan','manual','budget','speed-range','angle','power','reset','controller','team-file'])$(id).disabled=value;document.querySelectorAll('[data-target]').forEach(b=>b.disabled=value);$('execute').disabled=value||!planned;$('same-push').disabled=value||!rememberedPush||Math.hypot(...rememberedPush)>+$('speed-range').value+1e-12;}
function clearShot(){cancelAnimationFrame(animation);animation=0;planned=null;actual=null;frame=0;$('execute').disabled=true;$('pred-error').textContent=$('actual-error').textContent=$('gap').textContent='—';$('pred-caption').textContent='Distance to target center';$('actual-caption').textContent='Execute to reveal';$('gap-caption').textContent='Same push, two different paths';$('stage').textContent='01 / MAKE A PLAN';$('status').textContent='Ready when you are.';$('route-info').textContent='All models share the same push limit and 35-step duration. The score rewards the final landing, not the shortest route.';setBusy(busy);draw();}
function showPlan(result,ms){planned=result;rememberedPush=result.action.slice();actual=null;frame=0;$('pred-error').textContent=unit(result.assessment.error);$('pred-caption').textContent=result.assessment.hit?'Predicted to finish inside target':'Predicted to finish outside target';$('stage').textContent='02 / PREDICT THE OUTCOME';$('status').textContent=`${currentConfig.samePush?'Same push':currentConfig.manual?'Your push':`${currentConfig.budget} candidates`} · push ${result.speed.toFixed(3)} / limit ${currentConfig.maxSpeed.toFixed(3)} · ${steps} steps`;$('question').textContent='The dashed path is only a prediction. Where will the puck actually finish?';$('route-info').textContent=`Predicted route: ${routeLength(result.path).toFixed(2)} table widths. Long bouncing routes are allowed for every model. Execute to measure the actual route.`;setBusy(false);draw();}
function planShot(){if(busy)return;clearShot();currentConfig={start,target:target.slice(),steps,budget:budgets[+$('budget').value],maxSpeed:+$('speed-range').value,seed:123,controller:$('controller').value};setBusy(true);$('status').textContent='Searching the learned model…';worker.postMessage(currentConfig);}
function predictPush(action,samePush=false){
  const speed=Math.hypot(...action),maxSpeed=+$('speed-range').value;
  if(speed>maxSpeed+1e-12){$('status').textContent='This push exceeds the selected limit. Increase the limit or choose a weaker push.';return;}
  clearShot();
  try{
    const path=rollout(start,action,steps,s=>controllerStep(s,model,$('controller').value,parameters,improvedModel,teamModel,inferredModel));
    currentConfig={manual:!samePush,samePush,controller:$('controller').value,target:target.slice(),steps,maxSpeed};
    showPlan({action:action.slice(),path,speed,assessment:evaluatePath(path,target)},0);
  }catch(error){$('error').hidden=false;$('error').textContent=`Could not predict this push: ${error.message}`;setBusy(false);}
}
function manualShot(){
  if(busy)return;
  const angle=+$('angle').value*Math.PI/180,speed=+$('power').value/1000;
  predictPush([Math.cos(angle)*speed,Math.sin(angle)*speed]);
}
function samePush(){if(!busy&&rememberedPush)predictPush(rememberedPush.slice(),true);}
function syncPushLimit(){
  $('power').max=Math.round(+$('speed-range').value*1000);
  $('power').value=Math.min(+$('power').value,+$('power').max);
  $('power-value').textContent=(+$('power').value/1000).toFixed(3);
}
function finish(){
  const assessment=evaluatePath(actual,target),gap=distance(planned.path.at(-1),actual.at(-1));
  $('actual-error').textContent=unit(assessment.error);$('actual-caption').textContent=assessment.hit?'Finished inside target':'Finished outside target';$('gap').textContent=unit(gap);$('gap-caption').textContent='Distance between the two final positions';$('stage').textContent='03 / EXPLAIN THE DIFFERENCE';
  $('question').textContent=gap>.08?'The chosen shot behaved differently. Would fewer candidates, lower speed, or a different target change that?':'This shot roughly matched its prediction. Does that hold for a different search budget or target?';
  history.unshift({number:history.length+1,controller:currentConfig.controller,search:currentConfig.samePush?'Same push':currentConfig.manual?'Manual':currentConfig.budget,range:currentConfig.maxSpeed===.04?'Up to 0.040':'Up to 0.110',speed:Math.hypot(...planned.action),predictedTravel:routeLength(planned.path),actualTravel:routeLength(actual),pred:planned.assessment.error,actual:assessment.error,hit:assessment.hit,target:target.slice(),action:planned.action.slice()});
  $('history-count').textContent=`${history.length} shots`;$('history').innerHTML=history.map(r=>`<tr><td>${r.number}</td><td>${controllerNames[r.controller]}</td><td>${r.target.map(v=>v.toFixed(2)).join(", ")}</td><td>${r.search}</td><td>${r.range}</td><td>${r.speed.toFixed(3)}</td><td>${r.predictedTravel.toFixed(2)} / ${r.actualTravel.toFixed(2)}</td><td>${unit(r.pred)}</td><td>${unit(r.actual)}</td><td class="${r.hit?'hit':'miss'}">${r.hit?'Target reached':'Missed target'}</td></tr>`).join('');
  $('route-info').textContent=`Route length: predicted ${routeLength(planned.path).toFixed(2)}, actual ${routeLength(actual).toFixed(2)} table widths. Push ${Math.hypot(...planned.action).toFixed(3)}; limit ${currentConfig.maxSpeed.toFixed(3)}; ${steps} steps.`;setBusy(false);$('execute').disabled=true;$('status').textContent='Switch models and choose Predict same push to compare the exact same shot.';
}
function execute(){
  if(busy||!planned)return;
  function stopOnError(error){
    cancelAnimationFrame(animation);animation=0;setBusy(false);
    $('stage').textContent='EXECUTION STOPPED';$('status').textContent='Playback failed. You can reset or try the shot again.';
    $('error').hidden=false;$('error').textContent=`Could not play this shot: ${error.message||String(error)}`;
  }
  try{
    actual=rollout(start,planned.action,steps,physicsStep);frame=0;setBusy(true);
    $('error').hidden=true;$('stage').textContent='EXECUTING / SAME PUSH, ACTUAL PHYSICS';
    $('status').textContent=`Playing step 0 / ${steps}`;draw();
    // Use one animation-frame clock. The first callback may predate a performance.now() call made in that frame.
    let began=null;
    function tick(now){
      try{
        if(began===null)began=now;
        frame=Math.max(0,Math.min(steps,Math.floor((now-began)/65)));
        $('status').textContent=`Playing step ${frame} / ${steps}`;draw();
        if(frame<steps)animation=requestAnimationFrame(tick);
        else{animation=0;finish();}
      }catch(error){stopOnError(error);}
    }
    animation=requestAnimationFrame(tick);
  }catch(error){stopOnError(error);}
}
function selectTarget(next){if(busy)return;target=next;clearShot();document.querySelectorAll('[data-target]').forEach(b=>b.classList.toggle('active',b.dataset.target===next.join(',')));}
canvas.addEventListener('click',event=>{
  if(busy)return;const r=canvas.getBoundingClientRect(),scale=Math.min(r.width/1000,r.height/820),ox=(r.width-1000*scale)/2,oy=(r.height-820*scale)/2;
  const x=((event.clientX-r.left-ox)/scale-140)/720,y=((event.clientY-r.top-oy)/scale-45)/720;
  if(x<.08||x>.92||y<.08||y>.92||Math.hypot(x-.5,y-.5)<OBSTACLE.r+TARGET_RADIUS){$('status').textContent='Place the target inside the walls and clear of the bumper.';return;}
  selectTarget([x,y]);
});
$('controller').addEventListener('change',()=>{clearShot();$('controller-note').textContent={'neural':'A neural network predicts every next state.','improved-neural':'Known geometry: the positions of the walls and the bumper are written into the model. It learns every response from data, with separate gliding, wall and bumper pieces and collisions in surface coordinates.','inferred-neural':'New approach: the walls and the bumper are found from the training data alone, then the same kind of model is built on those found numbers. The planner and execution simulator are unchanged.','team':'Your locally loaded neural weights drive these predictions. The planner and execution simulator are unchanged.','contact-repair':'Known position/contact geometry and fitted collision response. Neural velocity predictions remain in free motion.','calibrated':'Known equation structure; drag and restitution fitted from the same training observations. Not a neural network.'}[$('controller').value];});
$('plan').addEventListener('click',planShot);$('execute').addEventListener('click',execute);$('manual').addEventListener('click',manualShot);$('same-push').addEventListener('click',samePush);
$('budget').addEventListener('input',()=>{$('budget-value').textContent=budgets[+$('budget').value];clearShot();});$('speed-range').addEventListener('change',()=>{syncPushLimit();clearShot();});
$('angle').addEventListener('input',()=>{$('angle-value').textContent=$('angle').value+'°';});$('power').addEventListener('input',()=>{$('power-value').textContent=(+$('power').value/1000).toFixed(3);});
document.querySelectorAll('[data-target]').forEach(b=>b.addEventListener('click',()=>selectTarget(b.dataset.target.split(',').map(Number))));
$('reset').addEventListener('click',()=>{rememberedPush=null;$('controller').value='neural';$('controller-note').textContent='A neural network predicts every next state.';history=[];$('history-count').textContent='0 shots';$('history').innerHTML='<tr><td colspan="10">Plan a push, make a prediction, then execute it.</td></tr>';$('budget').value=2;$('budget-value').textContent=512;$('speed-range').value='.11';syncPushLimit();selectTarget([.83,.72]);$('question').textContent='Would trying more candidate shots make the robot more reliable?';});
function fail(error){$('error').hidden=false;$('error').textContent=error.message||String(error);$('loading').hidden=true;setBusy(false);$('plan').disabled=true;}
const hashOf=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');
// Host view only: the trained reference models. Missing files (the student site build) mean the student view.
async function loadHostAssets(){
  const responses=await Promise.all(['calibration','neural-benchmark-summary','local-model','local-metadata'].map(n=>fetch(`../assets/${n}.json`)));
  if(responses.some(r=>!r.ok))return false;
  parameters=validateCalibration((await responses[0].json()).parameters);
  const comparison=await responses[1].json(),improvedMeta=await responses[3].json(),improvedBytes=await responses[2].arrayBuffer(),improvedSHA=await hashOf(improvedBytes);
  if(improvedSHA!==improvedMeta.sha256||improvedSHA!==comparison.improved_model_sha256)throw Error('Improved model or benchmark fingerprint mismatch.');
  improvedModel=validateImprovedModel(JSON.parse(new TextDecoder().decode(improvedBytes)));
  const extra=await Promise.all(['inferred-model','inferred-results'].map(n=>fetch(`../assets/${n}.json`)));
  if(extra.every(r=>r.ok)){
    const info=await extra[1].json(),bytes=await extra[0].arrayBuffer();
    if(await hashOf(bytes)!==info.model_sha256)throw Error('New-approach model fingerprint mismatch.');
    inferredModel=validateInferredModel(JSON.parse(new TextDecoder().decode(bytes)));
  }else $('controller').querySelector('option[value="inferred-neural"]')?.remove();
  return true;
}
function useStudentView(){document.documentElement.dataset.mode='student';for(const v of ['improved-neural','inferred-neural','calibrated'])$('controller').querySelector(`option[value="${v}"]`)?.remove();}
async function load(){
  try{
    const responses=await Promise.all(['model','metadata'].map(n=>fetch(`../assets/${n}.json`)));for(const r of responses)if(!r.ok)throw Error('Could not load trained model assets.');
    const bytes=await responses[0].arrayBuffer(),meta=await responses[1].json();if(await hashOf(bytes)!==meta.modelSHA256)throw Error('Model fingerprint mismatch.');
    model=validateModel(JSON.parse(new TextDecoder().decode(bytes)));
    if(!(hostMode&&await loadHostAssets()))useStudentView();
    worker=new Worker('./worker.mjs',{type:'module'});worker.onerror=e=>fail(Error(e.message||'Planner worker failed.'));worker.onmessage=e=>{if(e.data.error){fail(Error(e.data.error));return;}if(e.data.ready){$('loading').hidden=true;$('workspace').hidden=false;$('plan').disabled=false;draw();}else showPlan(e.data.result,e.data.milliseconds);};worker.postMessage({model,parameters,improvedModel,inferredModel,teamModel});
  }catch(error){fail(error);}
}
$('team-file').addEventListener('change',async event=>{
  const file=event.target.files[0];if(!file||busy)return;
  setBusy(true);
  try{
    if(file.size>5*1024*1024)throw Error('Choose a neural checkpoint smaller than 5 MB.');
    const candidate=validateTeamModel(JSON.parse(await file.text()));
    teamModel=candidate;$('team-option').disabled=false;$('controller').value='team';
    $('team-file-note').textContent=`Loaded ${file.name}. Stored in this browser tab only.`;
    $('controller-note').textContent='Your trained network is selected. Find a shot, then execute it to compare prediction with motion.';
    $('error').hidden=true;clearShot();worker.postMessage({model,parameters,improvedModel,inferredModel,teamModel});
  }catch(error){$('error').hidden=false;$('error').textContent=`Could not load team model: ${error.message}`;}
  finally{setBusy(false);event.target.value='';}
});
load();
