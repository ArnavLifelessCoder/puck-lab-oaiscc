"""Score several models and print ONE table: hits and mean miss on the 96 held-out scenes (and the 24 development scenes with --dev),
plus, for every model after the first, how many scenes it fixed or lost compared with the first (the baseline).

Each model is a team checkpoint (.json) or a submission file with a Repair class (.py), exactly as host/score_submission.py accepts.
Run from the puck-lab folder:

    python host/show_scores.py scratch/baseline.json scratch/candidate.json
    python host/show_scores.py host/local_neural_repair.py scratch/geofree/repair_local_inferred.py --dev

About 20 seconds per model per set. Nothing is written next to your files. Score once and do not tune afterwards.
"""
import argparse,importlib.util,json,sys,tempfile
from pathlib import Path
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'student'));sys.path.insert(0,str(ROOT/'host'))
from check_submission import evaluate

def repair_class(path):
    path=Path(path).resolve()
    if not path.is_file():sys.exit(f'Not found: {path}')
    if path.suffix.lower()=='.json':
        from neural_model import NeuralDynamics
        class Candidate:
            def __init__(self,neural,training_states,training_next_states):self.model=NeuralDynamics(path)
            def __call__(self,states):return self.model(states)
        return Candidate
    sys.path.insert(0,str(path.parent));sys.modules.pop('neural_model',None)
    spec=importlib.util.spec_from_file_location('submitted_'+path.stem.replace('-','_'),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.Repair

def score(path,scenes):
    with tempfile.TemporaryDirectory() as work:
        with threadpool_limits(limits=2):
            report=evaluate(scenes,repair_class(path),Path(work)/'report.json')
    return report['rows']

def label(path):
    p=Path(path);return (p.parent.name+'/'+p.stem) if p.parent.name not in ('.','') else p.stem

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('models',nargs='+',help='checkpoint .json or submission .py; the first is the baseline')
    ap.add_argument('--dev',action='store_true',help='also score the 24 development scenes')
    args=ap.parse_args()
    held=ROOT/'host/local-neural-confirmation-scenes.json'
    results=[]
    for m in args.models:
        print(f'scoring {label(m)} ...',flush=True)
        r96=score(m,held);r24=score(m,None) if args.dev else None
        results.append((label(m),r96,r24))
    base=results[0][1]
    mean=lambda rows:sum(x['actual_miss'] for x in rows)/len(rows)
    hits=lambda rows:sum(bool(x['hit']) for x in rows)
    head=f'{"model":38s} {"96 hits":>8s} {"96 miss":>8s}'+(f' {"24 hits":>8s} {"24 miss":>8s}' if args.dev else '')+f' {"fixed":>6s} {"lost":>5s}'
    print('\n'+head+'\n'+'-'*len(head))
    for i,(name,r96,r24) in enumerate(results):
        line=f'{name[:38]:38s} {hits(r96):>5d}/96 {mean(r96):>8.3f}'
        if args.dev:line+=f' {hits(r24):>5d}/24 {mean(r24):>8.3f}'
        if i==0:line+=f' {"(base)":>6s} {"":>5s}'
        else:
            fixed=sum(1 for a,b in zip(base,r96) if b['hit'] and not a['hit']);lost=sum(1 for a,b in zip(base,r96) if a['hit'] and not b['hit'])
            line+=f' {fixed:>6d} {lost:>5d}'
        print(line)
    print('\nmiss = mean distance from the target (table widths); fixed/lost = scenes hit by this model but not the baseline, and the reverse, on the 96 scenes.')

if __name__=='__main__':main()
