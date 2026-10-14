export const OBSTACLE={x:.5,y:.5,r:.115};
export const TARGET_RADIUS=.065;
export function physicsStep(s){
  let [x,y,vx,vy]=s;x+=vx;y+=vy;vx*=.985;vy*=.985;
  if(x<.02){x=.04-x;vx=Math.abs(vx)*.85;}else if(x>.98){x=1.96-x;vx=-Math.abs(vx)*.85;}
  if(y<.02){y=.04-y;vy=Math.abs(vy)*.85;}else if(y>.98){y=1.96-y;vy=-Math.abs(vy)*.85;}
  const dx=x-.5,dy=y-.5,d=Math.hypot(dx,dy);
  if(d<OBSTACLE.r){const nx=d?dx/d:1,ny=d?dy/d:0;x=.5+nx*OBSTACLE.r;y=.5+ny*OBSTACLE.r;const dot=vx*nx+vy*ny;if(dot<0){vx-=1.85*dot*nx;vy-=1.85*dot*ny;}}
  return [x,y,vx,vy];
}
export function validateModel(m){
  if(JSON.stringify(m.architecture)!=='[4,48,48,4]'||m.activation!=='tanh')throw Error('Unexpected model architecture.');
  for(let i=0;i<3;i++){
    if(m.weights[i]?.length!==m.architecture[i]*m.architecture[i+1]||m.biases[i]?.length!==m.architecture[i+1])throw Error('Invalid layer shape.');
    if(![...m.weights[i],...m.biases[i]].every(Number.isFinite))throw Error('Non-finite model weights.');
  }
  for(const name of ['inputMean','inputScale','outputMean','outputScale'])if(m[name]?.length!==4||!m[name].every(Number.isFinite))throw Error('Invalid normalization.');
  if(![...m.inputScale,...m.outputScale].every(v=>v>0))throw Error('Invalid scale.');
  return m;
}
export function learnedStep(s,m){
  let a=s.map((v,i)=>(v-m.inputMean[i])/m.inputScale[i]);
  for(let layer=0;layer<3;layer++){
    const n=m.architecture[layer+1],out=m.biases[layer].slice(),w=m.weights[layer];
    for(let i=0;i<a.length;i++)for(let j=0;j<n;j++)out[j]+=a[i]*w[i*n+j];
    a=layer<2?out.map(Math.tanh):out;
  }
  return a.map((v,i)=>s[i]+v*m.outputScale[i]+m.outputMean[i]);
}
function finiteShape(value,size){return Array.isArray(value)&&value.length===size&&value.every(Number.isFinite);}
export function validateLocalModel(m){
  if(m?.format!=='local-neural-v3'||m.activation!=='relu'||JSON.stringify(m.architecture)!=='[3,32,32,4]'||m.outputScale!==.11)throw Error('Unexpected local neural architecture.');
  for(const [key,n] of [['freeWeight',12],['wallWeight',6],['skipWeight',16]])if(!finiteShape(m[key],n))throw Error('Invalid local neural head.');
  if(m.weights?.length!==3||m.biases?.length!==3)throw Error('Invalid local neural layers.');
  for(let i=0;i<3;i++)if(!finiteShape(m.weights[i],m.architecture[i]*m.architecture[i+1])||!finiteShape(m.biases[i],m.architecture[i+1]))throw Error('Invalid local neural weights.');
  return m;
}
function affine(input,weights,bias){
  const output=bias.slice(),n=output.length;
  for(let i=0;i<input.length;i++)for(let j=0;j<n;j++)output[j]+=input[i]*weights[i*n+j];
  return output;
}
export const DEFAULT_GEOMETRY={wall_lo:.02,wall_hi:.98,cx:.5,cy:.5,radius:.115};
export function localNeuralStep(s,m,g=DEFAULT_GEOMETRY){
  const scale=.11,q=[s[0]+s[2],s[1]+s[3]];
  let delta=affine([s[2]/scale,s[3]/scale,1],m.freeWeight,[0,0,0,0]);
  for(let axis=0;axis<2;axis++)for(const [side,sign] of [[g.wall_lo,1],[g.wall_hi,-1]]){
    const clearance=(q[axis]-side)*sign;
    if(clearance<0){const response=affine([clearance/scale,s[axis+2]*sign/scale,1],m.wallWeight,[0,0]);delta[axis]=response[0]*sign;delta[axis+2]=response[1]*sign;}
  }
  const dx=q[0]-g.cx,dy=q[1]-g.cy,r=Math.hypot(dx,dy);
  if(r<g.radius){
    const nx=r>=1e-12?dx/r:1,ny=r>=1e-12?dy/r:0;
    const f=[(r-g.radius)/scale,(s[2]*nx+s[3]*ny)/scale,(-s[2]*ny+s[3]*nx)/scale];
    let a=f;
    for(let i=0;i<3;i++){a=affine(a,m.weights[i],m.biases[i]);if(i<2)a=a.map(v=>Math.max(0,v));}
    const response=affine([...f,1],m.skipWeight,a);
    delta=[response[0]*nx-response[1]*ny,response[0]*ny+response[1]*nx,response[2]*nx-response[3]*ny,response[2]*ny+response[3]*nx];
  }
  return s.map((v,i)=>v+scale*delta[i]);
}
// The same model, routed with geometry that was inferred from the training data (stored in m.geometry).
export function validateInferredModel(m){
  const g=m?.geometry;
  if(![g?.wall_lo,g?.wall_hi,g?.cx,g?.cy,g?.radius].every(Number.isFinite)||g.wall_lo>=g.wall_hi||g.radius<=0)throw Error('Invalid inferred geometry.');
  return validateLocalModel(m);
}
export function inferredStep(s,m){return localNeuralStep(s,m,m.geometry);}
export function validateTeamModel(m){
  const dims=m?.architecture;
  if(m?.format!=='team-mlp-v1'||typeof m.geometry!=='boolean'||!Array.isArray(dims)||dims.length!==4||dims[0]!== (m.geometry?11:4)||dims[3]!==4||!dims.every(n=>Number.isInteger(n)&&n>0&&n<=256))throw Error('Use a team-mlp-v1 checkpoint with hidden layers up to 256 units.');
  if(m.freeWeight!==undefined&&!finiteShape(m.freeWeight,12))throw Error('Invalid learned free-motion head.');
  if(m.weights?.length!==3||m.biases?.length!==3)throw Error('Missing team model layers.');
  for(let i=0;i<3;i++)if(!finiteShape(m.weights[i],dims[i]*dims[i+1])||!finiteShape(m.biases[i],dims[i+1]))throw Error('Invalid team model weights.');
  for(const [key,n] of [['inputMean',dims[0]],['inputScale',dims[0]],['outputMean',4],['outputScale',4]])if(!finiteShape(m[key],n))throw Error('Invalid team model normalization.');
  if([...m.inputScale,...m.outputScale].some(v=>v<=0))throw Error('Team model scales must be positive.');
  return m;
}
export function teamStep(s,m){
  if(m.freeWeight){
    const qx=s[0]+s[2],qy=s[1]+s[3];
    if(qx>=.02&&qx<=.98&&qy>=.02&&qy<=.98&&Math.hypot(qx-.5,qy-.5)>=.115){
      const delta=affine([s[2]/.11,s[3]/.11,1],m.freeWeight,[0,0,0,0]);
      return s.map((v,i)=>v+.11*delta[i]);
    }
  }
  let f=s.slice();
  if(m.geometry){const qx=s[0]+s[2],qy=s[1]+s[3],dx=qx-.5,dy=qy-.5,r=Math.hypot(dx,dy);f.push(qx-.02,.98-qx,qy-.02,.98-qy,r-.115,dx/Math.max(r,1e-8),dy/Math.max(r,1e-8));}
  let a=f.map((v,i)=>(v-m.inputMean[i])/m.inputScale[i]);
  for(let i=0;i<3;i++){a=affine(a,m.weights[i],m.biases[i]);if(i<2)a=a.map(v=>Math.max(0,v));}
  return s.map((v,i)=>v+a[i]*m.outputScale[i]+m.outputMean[i]);
}
export function validateImprovedModel(m){
  if(m?.format==='local-neural-v3')return validateLocalModel(m);
  if(m?.format!=='gated-residual-mlp-v2'||m.activation!=='relu'||JSON.stringify(m.architecture)!=='[11,64,64,4]')throw Error('Unexpected improved network architecture.');
  const shape=(a,n)=>Array.isArray(a)&&a.length===n&&a.every(Number.isFinite);
  if(!Array.isArray(m.weights)||!Array.isArray(m.biases)||m.weights.length!==3||m.biases.length!==3)throw Error('Invalid improved network layers.');
  for(let i=0;i<3;i++)if(!shape(m.weights[i],m.architecture[i]*m.architecture[i+1])||!shape(m.biases[i],m.architecture[i+1]))throw Error('Invalid improved network weights.');
  for(const [key,n] of [['skipWeight',44],['skipBias',4],['freeWeight',16],['freeBias',4]])if(!shape(m[key],n))throw Error('Invalid improved network branch.');
  if(m.outputScale!==.11)throw Error('Invalid improved network output scale.');
  return m;
}
export function improvedStep(s,m){
  if(m.format==='local-neural-v3')return localNeuralStep(s,m);
  const [x,y,vx,vy]=s,qx=x+vx,qy=y+vy,dx=qx-.5,dy=qy-.5,r=Math.hypot(dx,dy);
  const f=[(x-.5)*2,(y-.5)*2,vx/.11,vy/.11,(qx-.02)/.11,(.98-qx)/.11,(qy-.02)/.11,(.98-qy)/.11,(r-.115)/.11,dx/Math.max(r,1e-8),dy/Math.max(r,1e-8)];
  let delta;
  // Geometry selects the branch. Every state change still comes from learned weights.
  if(!f.slice(4,9).some(v=>v<0)){
    delta=m.freeBias.slice();
    for(let i=0;i<4;i++)for(let j=0;j<4;j++)delta[j]+=f[i]*m.freeWeight[i*4+j];
  }else{
    let a=f;
    for(let layer=0;layer<3;layer++){
      const n=m.architecture[layer+1],out=m.biases[layer].slice(),w=m.weights[layer];
      for(let i=0;i<a.length;i++)for(let j=0;j<n;j++)out[j]+=a[i]*w[i*n+j];
      a=layer<2?out.map(v=>Math.max(0,v)):out;
    }
    delta=a.map((v,j)=>v+m.skipBias[j]);
    for(let i=0;i<11;i++)for(let j=0;j<4;j++)delta[j]+=f[i]*m.skipWeight[i*4+j];
  }
  return delta.map((v,i)=>s[i]+m.outputScale*v);
}
export function rollout(start,action,steps,next){
  let s=[...start,...action];const path=[s];
  for(let t=0;t<steps;t++){s=next(s);if(!s.every(Number.isFinite))throw Error('Model rollout became non-finite.');path.push(s);}
  return path;
}
export function distance(p,q){return Math.hypot(p[0]-q[0],p[1]-q[1]);}
export function routeLength(path){return path.slice(1).reduce((sum,s,i)=>sum+distance(s,path[i]),0);}
export function evaluatePath(path,target){
  let outside=0,obstacleSteps=0;
  for(const s of path){if(s[0]<.02||s[0]>.98||s[1]<.02||s[1]>.98)outside++;if(Math.hypot(s[0]-.5,s[1]-.5)<OBSTACLE.r-1e-7)obstacleSteps++;}
  const error=distance(path.at(-1),target);
  return {error,hit:error<=TARGET_RADIUS,outside,obstacleSteps};
}
export function validateCalibration(p){
  for(const k of ['drag','wall_restitution','bumper_restitution'])if(!Number.isFinite(p?.[k])||p[k]<0||p[k]>1)throw Error('Invalid calibrated coefficient.');
  return p;
}
export function structuredStep(s,p,m=null){
  let [x,y,vx,vy]=s;x+=vx;y+=vy;vx*=p.drag;vy*=p.drag;let contact=false;
  if(x<.02){x=.04-x;vx=-vx*p.wall_restitution;contact=true;}else if(x>.98){x=1.96-x;vx=-vx*p.wall_restitution;contact=true;}
  if(y<.02){y=.04-y;vy=-vy*p.wall_restitution;contact=true;}else if(y>.98){y=1.96-y;vy=-vy*p.wall_restitution;contact=true;}
  const dx=x-.5,dy=y-.5,d=Math.hypot(dx,dy);
  if(d<OBSTACLE.r){contact=true;const nx=d?dx/d:1,ny=d?dy/d:0;x=.5+nx*OBSTACLE.r;y=.5+ny*OBSTACLE.r;const dot=vx*nx+vy*ny;if(dot<0){vx-=(1+p.bumper_restitution)*dot*nx;vy-=(1+p.bumper_restitution)*dot*ny;}}
  if(m&&!contact){const neural=learnedStep(s,m);vx=neural[2];vy=neural[3];}
  return [x,y,vx,vy];
}
export function controllerStep(s,m,controller='neural',parameters=null,improvedModel=null,teamModel=null,inferredModel=null){
  if(controller==='neural')return learnedStep(s,m);
  if(controller==='team')return teamStep(s,teamModel);
  if(controller==='improved-neural')return improvedStep(s,improvedModel);
  if(controller==='inferred-neural')return inferredStep(s,inferredModel);
  if(!['contact-repair','calibrated'].includes(controller))throw Error('Unknown controller.');
  return structuredStep(s,parameters,controller==='contact-repair'?m:null);
}
export function plan(m,{start,target,steps=35,budget=256,maxSpeed=.11,seed=123,controller='neural',parameters=null,improvedModel=null,teamModel=null,inferredModel=null}){
  if(controller==='improved-neural')validateImprovedModel(improvedModel);
  else if(controller==='inferred-neural')validateInferredModel(inferredModel);
  else if(controller==='team')validateTeamModel(teamModel);
  else if(controller!=='neural')validateCalibration(parameters);

  let state=seed>>>0;
  const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296;};
  let best=null;
  for(let i=0;i<budget;i++){
    const angle=random()*Math.PI*2,speed=.008+random()*(maxSpeed-.008),action=[Math.cos(angle)*speed,Math.sin(angle)*speed];
    const path=rollout(start,action,steps,s=>controllerStep(s,m,controller,parameters,improvedModel,teamModel,inferredModel)),assessment=evaluatePath(path,target);
    // Rank by the selected prediction model. The execution simulator is not queried here.
    const cost=assessment.error+assessment.outside*.025+assessment.obstacleSteps*.025;
    if(!best||cost<best.cost)best={action,path,cost,assessment,speed,angle};
  }
  return best;
}
