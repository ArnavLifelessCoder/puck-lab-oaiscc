from pathlib import Path
import sys,json
import numpy as np
from threadpoolctl import threadpool_limits
PROJECT=Path(__file__).resolve().parents[1]
for folder in ['student','host','training']:sys.path.insert(0,str(PROJECT/folder))
from lab import Dynamics,physics
from controllers import contacts,structured_step
from train import generate
ROOT=Path(__file__).resolve().parents[1]
x,delta=generate(np.random.default_rng(72),160);truth=x+delta
_,walls,bumper,_=contacts(x);contact=np.any(walls,axis=1)|bumper
model=Dynamics()
with threadpool_limits(limits=2):pred=model(x)
def metrics(mask):
 error=pred[mask]-truth[mask]
 return dict(count=int(mask.sum()),position_rmse=float(np.sqrt(np.mean(error[:,:2]**2))),velocity_rmse=float(np.sqrt(np.mean(error[:,2:]**2))))
report=dict(free=metrics(~contact),contact=metrics(contact),all=metrics(np.ones(len(x),dtype=bool)),interpretation='Conditional prediction errors measured on the original independent validation trajectories. Contact is identified from geometry, never from prediction error. This does not by itself establish contact error as the cause of every control failure.')
(ROOT/'host/results/diagnosis.json').write_text(json.dumps(report,indent=2))
parameters=json.loads((ROOT/'assets/calibration.json').read_text())['parameters']
# Include contact-rich fixtures, not just mostly free-motion random examples.
ids=np.r_[np.flatnonzero(~contact)[:8],np.flatnonzero(contact)[:24]];states=x[ids]
with threadpool_limits(limits=2):
 fixtures=dict(states=states.tolist(),calibrated=structured_step(states,parameters).tolist(),contactRepair=structured_step(states,parameters,model).tolist())
(ROOT/'assets/controller-parity.json').write_text(json.dumps(fixtures))
np.testing.assert_allclose(structured_step(x,parameters),np.array([physics(s) for s in x]),atol=1e-12,rtol=0)
print(json.dumps(report,indent=2))
