"""Train a residual neural dynamics model. Geometry is input, never a response override."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'host'));sys.path.insert(0,str(ROOT/'student'))
from controllers import structured_step
SEED=173
PARAMS=dict(drag=.985,wall_restitution=.85,bumper_restitution=.85)

def features(states):
    q=states[:,:2]+states[:,2:]
    d=q-.5;r=np.linalg.norm(d,axis=1,keepdims=True)
    normal=d/np.maximum(r,1e-8)
    return np.column_stack([(states[:,:2]-.5)*2,states[:,2:]/.11,(q[:,0]-.02)/.11,(.98-q[:,0])/.11,(q[:,1]-.02)/.11,(.98-q[:,1])/.11,(r[:,0]-.115)/.11,normal]).astype(np.float32)

def dataset(seed,count):
    rng=np.random.default_rng(seed)
    all_states=[]
    for mode in range(3):
        n=count//3
        p=rng.uniform(.02,.98,(n,2))
        if mode==1:
            axis=rng.integers(0,2,n);side=rng.integers(0,2,n)
            p[np.arange(n),axis]=np.where(side,.98-rng.uniform(0,.12,n),.02+rng.uniform(0,.12,n))
        if mode==2:
            angle=rng.uniform(0,2*np.pi,n);radius=rng.uniform(.115,.24,n)
            p=.5+np.column_stack([np.cos(angle),np.sin(angle)])*radius[:,None]
        valid=np.linalg.norm(p-.5,axis=1)>=.115
        p=p[valid];n=len(p)
        angle=rng.uniform(0,2*np.pi,n);speed=rng.uniform(0,.115,n)
        v=np.column_stack([np.cos(angle),np.sin(angle)])*speed[:,None]
        all_states.append(np.column_stack([p,v]))
    states=np.concatenate(all_states)
    next_states=structured_step(states,PARAMS)
    return states,next_states

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.skip=nn.Linear(11,4)
        self.layers=nn.Sequential(nn.Linear(11,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,4))
    def forward(self,x):return self.skip(x)+self.layers(x)

def export(model,path):
    layers=[model.layers[i] for i in [0,2,4]]
    value=dict(format='residual-geometry-mlp-v1',architecture=[11,64,64,4],activation='relu',weights=[l.weight.detach().numpy().T.ravel().tolist() for l in layers],biases=[l.bias.detach().numpy().tolist() for l in layers],skipWeight=model.skip.weight.detach().numpy().T.ravel().tolist(),skipBias=model.skip.bias.detach().numpy().tolist(),outputScale=.11)
    path.write_text(json.dumps(value,separators=(',',':'),allow_nan=False))

def main():
    torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    s,next_s=dataset(SEED,90000);vs,vnext=dataset(SEED+1,9000)
    x=torch.from_numpy(features(s));y=torch.from_numpy(((next_s-s)/.11).astype(np.float32))
    vx=torch.from_numpy(features(vs));vy=torch.from_numpy(((vnext-vs)/.11).astype(np.float32))
    model=Network();opt=torch.optim.Adam(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[100,150],gamma=.3)
    logs=[];began=time.time()
    for epoch in range(180):
        model.train();order=torch.randperm(len(x));total=0
        for ids in order.split(1024):
            pred=model(x[ids]);loss=(pred-y[ids]).square().mean()
            opt.zero_grad();loss.backward();opt.step();total+=loss.item()*len(ids)
        scheduler.step();model.eval()
        with torch.no_grad():val=(model(vx)-vy).square().mean().item()
        logs.append(dict(epoch=epoch+1,training_mse=total/len(x),validation_mse=val))
        if epoch%20==0 or epoch==179:print(f'Epoch {epoch+1}/180 train={total/len(x):.6f} validation={val:.6f}',flush=True)
    destination=ROOT/'host/experiments/dense-neural-v1/improved-model.json';destination.parent.mkdir(parents=True,exist_ok=True);export(model,destination)
    with torch.no_grad():pred=vs+model(vx).numpy()*.11
    meta=dict(seed=SEED,validation_seed=SEED+1,training_samples=len(s),validation_samples=len(vs),epochs=180,architecture=[11,64,64,4],position_rmse=float(np.sqrt(np.mean((pred[:,:2]-vnext[:,:2])**2))),velocity_rmse=float(np.sqrt(np.mean((pred[:,2:]-vnext[:,2:])**2))),torch=torch.__version__,numpy=np.__version__,seconds=time.time()-began,logs=logs,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),training_state_sha256=hashlib.sha256(s.tobytes()).hexdigest(),validation_state_sha256=hashlib.sha256(vs.tobytes()).hexdigest(),protocol='Fixed 180 epochs, independent one-step training/validation samples. All outputs are neural residual predictions. Inputs include raw state and geometric clearances/normal; no analytic collision response or simulator call in inference. Training has additional near-wall/bumper states and speeds 0..0.115. Extra data, features, architecture and compute differ from baseline, so this is a combined repair, not a single-factor ablation. No confirmation evaluation used for selection.')
    (destination.parent/'improved-metadata.json').write_text(json.dumps(meta,indent=2))
    ids=np.linspace(0,len(vs)-1,48,dtype=int)
    (destination.parent/'improved-parity.json').write_text(json.dumps(dict(states=vs[ids].tolist(),nextStates=pred[ids].tolist())))
    print(json.dumps({k:v for k,v in meta.items() if k not in ['logs','protocol']},indent=2))
if __name__=='__main__':main()
