from pathlib import Path
import sys,json,hashlib
import numpy as np
PROJECT=Path(__file__).resolve().parents[1]
for folder in ['student','host','training']:sys.path.insert(0,str(PROJECT/folder))
from controllers import fit_parameters,structured_step
from train import generate,SEED
ROOT=Path(__file__).resolve().parents[1]
x,delta=generate(np.random.default_rng(SEED),1200)
parameters,counts=fit_parameters(x,x+delta)
export=dict(parameters=parameters,training_seed=SEED,training_samples=len(x),regime_counts=counts,knowledge='Known position integration, table and bumper geometry, contact normal response. Only drag and restitution are fitted. This is a calibrated physical model, not another neural network.',training_sha256=hashlib.sha256(x.tobytes()+delta.tobytes()).hexdigest())
(ROOT/'assets/calibration.json').write_text(json.dumps(export,indent=2))
prediction=structured_step(x,parameters)
print(json.dumps(export,indent=2))
print('Training maximum state error:',np.max(np.abs(prediction-(x+delta))))
