"""One named neural experiment: train, evaluate, compare, preserve artifacts."""
from pathlib import Path
import argparse, json, re, subprocess, sys, hashlib
from threadpoolctl import threadpool_limits
from neural_model import NeuralDynamics
from check_submission import evaluate
HERE=Path(__file__).resolve().parent


def name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,47}',value):
        raise argparse.ArgumentTypeError('Use 1-48 letters, digits, underscores or hyphens.')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',type=name,required=True)
    parser.add_argument('--compare',type=name)
    parser.add_argument('--data',choices=['original','expanded'],default='original')
    parser.add_argument('--geometry',action='store_true')
    parser.add_argument('--free-head',action='store_true')
    parser.add_argument('--contact-weight',type=float,default=1)
    parser.add_argument('--width',type=int,default=48)
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();run=HERE/'results/runs'/args.name
    if run.exists():parser.error('That experiment already exists. Choose a new name to preserve earlier evidence.')
    comparison=HERE/'results/runs'/args.compare/'score.json' if args.compare else None
    if comparison and not comparison.is_file():parser.error('The comparison run has no completed score.')
    run.mkdir(parents=True);checkpoint=run/'network.json'
    command=[sys.executable,str(HERE/'train_repair.py'),'--output',str(checkpoint),'--data',args.data,'--contact-weight',str(args.contact_weight),'--width',str(args.width),'--epochs',str(args.epochs),'--seed',str(args.seed)]
    if args.geometry:command.append('--geometry')
    if args.free_head:command.append('--free-head')
    subprocess.run(command,check=True,cwd=HERE)
    class Candidate:
        def __init__(self,neural,training_states,training_next_states):self.model=NeuralDynamics(checkpoint)
        def __call__(self,states):return self.model(states)
    with threadpool_limits(limits=2):report=evaluate(repair_class=Candidate,output_path=run/'score.json')
    report['configuration']={k:v for k,v in vars(args).items() if k not in ['name','compare']}
    report['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report['evaluation_sha256']=hashlib.sha256((HERE/'check_submission.py').read_bytes()+(HERE/'lab.py').read_bytes()+(HERE.parent/'assets/development-scenes.json').read_bytes()).hexdigest()
    report['mean_miss']=sum(r['actual_miss'] for r in report['rows'])/report['total']
    print(f"\n{args.name}: {report['hits']}/{report['total']} targets, mean miss {report['mean_miss']:.4f}")
    if comparison:
        before=json.loads(comparison.read_text())
        if before.get('evaluation_sha256')!=report['evaluation_sha256']:raise ValueError('Evaluator or scenes differ; these runs are not directly comparable.')
        changes={k:[before['configuration'].get(k,False),v] for k,v in report['configuration'].items() if before['configuration'].get(k,False)!=v}
        fixed=[r['scene'] for a,r in zip(before['rows'],report['rows']) if r['hit'] and not a['hit']]
        lost=[r['scene'] for a,r in zip(before['rows'],report['rows']) if a['hit'] and not r['hit']]
        report['comparison']=dict(name=args.compare,changed_settings=changes,new_successes=fixed,new_failures=lost)
        print(f"Before: {before['hits']}/{before['total']}. New successes: {fixed}. New failures: {lost}.")
        print('Changed settings:',json.dumps(changes))
    (run/'score.json').write_text(json.dumps(report,indent=2))
    print('Saved run:',run)
    print('Load network.json in the website to try your actual trained model. Submit this run folder with your edited training/inference code.')

if __name__=='__main__':main()
