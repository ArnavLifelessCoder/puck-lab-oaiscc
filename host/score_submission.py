"""Run once after submission freeze. Review submitted code before executing it."""
from pathlib import Path
import sys,importlib.util
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'student'))
sys.path.insert(0,str(ROOT/'host'))
from check_submission import evaluate
if len(sys.argv)!=2:raise SystemExit('Usage: python host/score_submission.py path/to/network.json OR path/to/repair.py')
submission=Path(sys.argv[1]).resolve()
if submission.suffix.lower()=='.json':
    from neural_model import NeuralDynamics
    class Candidate:
        def __init__(self,neural,training_states,training_next_states):self.model=NeuralDynamics(submission)
        def __call__(self,states):return self.model(states)
    repair_class=Candidate
else:
    sys.path.insert(0,str(submission.parent))
    sys.modules.pop('neural_model',None)
    spec=importlib.util.spec_from_file_location('submitted_repair',submission)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    repair_class=module.Repair
with threadpool_limits(limits=2):
    evaluate(ROOT/'host/local-neural-confirmation-scenes.json',repair_class,submission.parent/(submission.stem+'-scored.json'))
