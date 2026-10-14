from pathlib import Path
import sys,json,time,math,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
for folder in ['student','host']:sys.path.insert(0,str(ROOT/folder))
from lab import Dynamics,candidates
from improved_dynamics import ImprovedDynamics
from local_dynamics import LocalDynamics
from controllers import batch_rollout,costs,scene_set,structured_step
mode=sys.argv[1] if len(sys.argv)>1 else 'development'
if mode not in ['development','confirmation']:raise ValueError('Unknown partition')
protocol=json.loads((ROOT/'host/local-neural-protocol.json').read_text())
scenes=json.loads((ROOT/'assets/development-scenes.json').read_text()) if mode=='development' else scene_set(protocol['confirmation_seed'],protocol['confirmation_count'])
actions=candidates(protocol['budget'],protocol['max_speed'],protocol['candidate_seed'])
models={'neural':Dynamics(),'previous-neural':ImprovedDynamics(),'improved-neural':LocalDynamics()}
model_hash=hashlib.sha256((ROOT/'assets/local-model.json').read_bytes()).hexdigest()
if mode=='confirmation':
    frozen=json.loads((ROOT/'host/local-neural-freeze.json').read_text())
    if frozen['sha256']!=model_hash:raise ValueError('Checkpoint changed after freeze')
    for name,file in [('previous_sha256','improved-model.json'),('original_sha256','model.json')]:
        if frozen[name]!=hashlib.sha256((ROOT/'assets'/file).read_bytes()).hexdigest():raise ValueError('Baseline changed after freeze')
rows=[]
with threadpool_limits(limits=2):
 for i,scene in enumerate(scenes):
  for name,transition in models.items():
   t=time.perf_counter();paths=batch_rollout(scene['start'],actions,transition,protocol['horizon']);selected=int(np.argmin(costs(paths,scene['target'])))
   planning=time.perf_counter()-t
   # Ground truth is evaluated only after the action index is fixed.
   truth=batch_rollout(scene['start'],actions[selected:selected+1],lambda s:structured_step(s,dict(drag=.985,wall_restitution=.85,bumper_restitution=.85)),protocol['horizon'])[0]
   miss=float(np.linalg.norm(truth[-1,:2]-scene['target']))
   rows.append(dict(scene=i,controller=name,selected=selected,action=actions[selected].tolist(),predicted_miss=float(np.linalg.norm(paths[selected,-1,:2]-scene['target'])),actual_miss=miss,hit=miss<=protocol['target_radius'],seconds=planning))
  if i%12==0:print(f'{mode}: {i+1}/{len(scenes)}',flush=True)
summary={};base={r['scene']:r for r in rows if r['controller']=='neural'}
for name in models:
 r=[v for v in rows if v['controller']==name];n=len(r);k=sum(v['hit'] for v in r);phat=k/n;z=1.96;den=1+z*z/n;c=(phat+z*z/(2*n))/den;h=z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den
 summary[name]=dict(hits=k,total=n,success=k/n,wilson95=[max(0,c-h),min(1,c+h)],mean_miss=float(np.mean([v['actual_miss'] for v in r])),median_planning_seconds=float(np.median([v['seconds'] for v in r])),corrected=[v['scene'] for v in r if v['hit'] and not base[v['scene']]['hit']],regressed=[v['scene'] for v in r if not v['hit'] and base[v['scene']]['hit']])
report=dict(partition=mode,improved_model_sha256=model_hash,protocol=protocol,scenes=scenes,summary=summary,rows=rows,limitation='One trained checkpoint per method in a deterministic simulator. New and previous improved models use the same training observations. Contact coordinate representation, head structure, training seed and optimization changed. The original model uses less data. No claim of single-factor causation or general robot reliability.')
(ROOT/'host/results'/f'local-neural-{mode}.json').write_text(json.dumps(report,indent=2))
print(json.dumps(summary,indent=2))
