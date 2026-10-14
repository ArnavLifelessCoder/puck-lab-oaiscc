"""Run a submitted next-state model on all fixed development scenes."""
from pathlib import Path
import json,time
import numpy as np
from lab import Dynamics,candidates,predicted_paths,rollout,physics
from repair import Repair
ROOT=Path(__file__).resolve().parents[1]
RESULTS=Path(__file__).resolve().parent/'results'

def evaluate(scenes_path=None, repair_class=Repair, output_path=None):
    with np.load(ROOT/'assets/training-observations.npz',allow_pickle=False) as d:
        repaired=repair_class(Dynamics(),d['states'],d['next_states'])
    scenes=json.loads((Path(scenes_path) if scenes_path else ROOT/'assets/development-scenes.json').read_text())
    actions=candidates(2048,.11,123);rows=[]
    for i,scene in enumerate(scenes):
        started=time.perf_counter()
        paths=predicted_paths(repaired,scene['start'],actions,35)
        if paths.shape!=(2048,36,4) or not np.isfinite(paths).all():raise ValueError('Invalid predicted state tensor')
        miss=np.linalg.norm(paths[:,-1,:2]-scene['target'],axis=1)
        outside=np.any((paths[:,:,:2]<.02)|(paths[:,:,:2]>.98),axis=2).sum(axis=1)
        inside=(np.linalg.norm(paths[:,:,:2]-.5,axis=2)<.115-1e-7).sum(axis=1)
        chosen=int(np.argmin(miss+.025*(outside+inside)))
        actual=rollout(scene['start'],actions[chosen],35,physics)
        error=float(np.linalg.norm(actual[-1,:2]-scene['target']))
        rows.append(dict(scene=i,hit=error<=.065,actual_miss=error,predicted_miss=float(miss[chosen]),seconds=time.perf_counter()-started))
    report=dict(partition='confirmation' if scenes_path else 'development',hits=sum(r['hit'] for r in rows),total=len(rows),rows=rows)
    RESULTS.mkdir(exist_ok=True)
    (Path(output_path) if output_path else RESULTS/'submission-results.json').write_text(json.dumps(report,indent=2))
    print(f"{report['partition']}: {report['hits']}/{report['total']} targets.")
    return report

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();candidate=Repair
    if args.checkpoint:
        from neural_model import NeuralDynamics
        class Candidate:
            def __init__(self,neural,training_states,training_next_states):self.model=NeuralDynamics(args.checkpoint)
            def __call__(self,states):return self.model(states)
        candidate=Candidate
    with threadpool_limits(limits=2):evaluate(repair_class=candidate,output_path=args.output)
