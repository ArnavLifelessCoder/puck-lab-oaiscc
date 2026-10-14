from pathlib import Path
import sys,json,time,math
import numpy as np
from threadpoolctl import threadpool_limits
PROJECT=Path(__file__).resolve().parents[1]
for folder in ['student','host','training']:sys.path.insert(0,str(PROJECT/folder))
from lab import Dynamics,candidates
from controllers import transitions,batch_rollout,costs,scene_set
ROOT=Path(__file__).resolve().parents[1]
protocol=json.loads((ROOT/'host/protocol.json').read_text())
parameters=json.loads((ROOT/'assets/calibration.json').read_text())['parameters']
mode=sys.argv[1] if len(sys.argv)>1 else 'development'
if mode not in ['development','confirmation']:raise ValueError('Unknown partition')
scenes=scene_set(protocol[mode+'_seed'],protocol[mode+'_scenes'])
models=transitions(Dynamics(),parameters)
actions=candidates(protocol['budget'],protocol['max_speed'],protocol['candidate_seed'])
rows=[]
with threadpool_limits(limits=2):
 for i,scene in enumerate(scenes):
  truth=batch_rollout(scene['start'],actions,models['oracle'],protocol['horizon'])
  for name,transition in models.items():
   started=time.perf_counter();paths=truth if name=='oracle' else batch_rollout(scene['start'],actions,transition,protocol['horizon']);selected=int(np.argmin(costs(paths,scene['target'])))
   predicted=float(np.linalg.norm(paths[selected,-1,:2]-scene['target']));actual=float(np.linalg.norm(truth[selected,-1,:2]-scene['target']))
   rows.append(dict(scene=i,controller=name,selected=selected,predicted_miss=predicted,actual_miss=actual,hit=actual<=protocol['target_radius'],seconds=time.perf_counter()-started))
  if i%12==0:print(f'{mode}: {i+1}/{len(scenes)}',flush=True)
def wilson(k,n):
 z=1.96;p=k/n;den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den;return [max(0,c-h),min(1,c+h)]
summary={}
baseline={r['scene']:r for r in rows if r['controller']=='neural'}
for name in models:
 r=[v for v in rows if v['controller']==name];k=sum(v['hit'] for v in r)
 summary[name]=dict(hits=k,total=len(r),success=k/len(r),wilson95=wilson(k,len(r)),mean_miss=float(np.mean([v['actual_miss'] for v in r])),median_planning_seconds=float(np.median([v['seconds'] for v in r])) if name!='oracle' else None,corrected=[v['scene'] for v in r if v['hit'] and not baseline[v['scene']]['hit']],regressed=[v['scene'] for v in r if not v['hit'] and baseline[v['scene']]['hit']])
report=dict(partition=mode,protocol=protocol,scenes=scenes,summary=summary,rows=rows,limitation='Synthetic deterministic environment with known geometry. Calibration matches the simulator family by design. One checkpoint and environment; Wilson intervals summarize sampled scenes, not model-seed uncertainty. Oracle uses privileged true dynamics.')
(ROOT/'host/results'/f'{mode}-results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(summary,indent=2))
