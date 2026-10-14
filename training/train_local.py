"""Train a contact-local neural predictor on the same observations as v2."""
from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'student'))
from local_dynamics import LocalDynamics


def local_examples(s,next_s):
    q=s[:,:2]+s[:,2:];delta=(next_s-s)/.11
    r=np.linalg.norm(q-.5,axis=1)
    free=((q>=.02)&(q<=.98)).all(1)&(r>=.115)
    free_x=np.column_stack([s[free,2:]/.11,np.ones(free.sum())])
    wx=[];wy=[]
    for axis in range(2):
        for side,sign in [(.02,1),(.98,-1)]:
            clearance=(q[:,axis]-side)*sign;mask=clearance<0
            wx.append(np.column_stack([clearance[mask]/.11,s[mask,axis+2]*sign/.11,np.ones(mask.sum())]))
            wy.append(delta[mask][:,[axis,axis+2]]*sign)
    mask=r<.115
    normal=(q[mask]-.5)/np.maximum(r[mask,None],1e-12);normal[r[mask]<1e-12]=[1,0]
    tangent=np.column_stack([-normal[:,1],normal[:,0]])
    x=np.column_stack([(r[mask]-.115)/.11,np.sum(s[mask,2:]*normal,axis=1)/.11,np.sum(s[mask,2:]*tangent,axis=1)/.11])
    y=np.column_stack([np.sum(delta[mask,:2]*normal,axis=1),np.sum(delta[mask,:2]*tangent,axis=1),np.sum(delta[mask,2:]*normal,axis=1),np.sum(delta[mask,2:]*tangent,axis=1)])
    return free_x,delta[free],np.concatenate(wx),np.concatenate(wy),x,y


def main():
    seed=373;torch.set_num_threads(2);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    with np.load(ROOT/'assets/neural-training.npz',allow_pickle=False) as d:
        s,n,vs,vn=[d[k] for k in ['states','next_states','validation_states','validation_next_states']]
    fx,fy,wx,wy,x,y=local_examples(s,n);*_,vx,vy=local_examples(vs,vn)
    free=np.linalg.lstsq(fx,fy,rcond=None)[0];wall=np.linalg.lstsq(wx,wy,rcond=None)[0]
    skip=np.linalg.lstsq(np.column_stack([x,np.ones(len(x))]),y,rcond=None)[0]
    residual=y-np.column_stack([x,np.ones(len(x))])@skip
    target=torch.tensor(residual,dtype=torch.float32);tx=torch.tensor(x,dtype=torch.float32)
    model=nn.Sequential(nn.Linear(3,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,4))
    nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,[500,800],gamma=.3)
    logs=[];began=time.perf_counter()
    for epoch in range(1000):
        for ids in torch.randperm(len(x)).split(512):
            loss=(model(tx[ids])-target[ids]).square().mean();optimizer.zero_grad();loss.backward();optimizer.step()
        schedule.step()
        if epoch%100==0 or epoch==999:
            with torch.no_grad(): pred=model(torch.tensor(vx,dtype=torch.float32)).numpy()+np.column_stack([vx,np.ones(len(vx))])@skip
            error=float(np.mean((pred-vy)**2));logs.append(dict(epoch=epoch+1,validation_local_mse=error))
            print(f'Epoch {epoch+1}/1000: local contact validation MSE={error:.9f}',flush=True)
    layers=[model[i] for i in [0,2,4]]
    checkpoint=dict(format='local-neural-v3',architecture=[3,32,32,4],activation='relu',outputScale=.11,
                    freeWeight=free.ravel().tolist(),wallWeight=wall.ravel().tolist(),skipWeight=skip.ravel().tolist(),
                    weights=[l.weight.detach().numpy().T.ravel().tolist() for l in layers],biases=[l.bias.detach().numpy().tolist() for l in layers])
    path=ROOT/'assets/local-model.json';path.write_text(json.dumps(checkpoint,separators=(',',':'),allow_nan=False))
    prediction=LocalDynamics(path)(vs)
    metadata=dict(seed=seed,training_samples=len(s),validation_samples=len(vs),bumper_training_samples=len(x),wall_training_samples=len(wx),epochs=1000,
                  seconds=time.perf_counter()-began,logs=logs,torch=torch.__version__,numpy=np.__version__,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  training_state_sha256=hashlib.sha256(s.tobytes()).hexdigest(),validation_state_sha256=hashlib.sha256(vs.tobytes()).hexdigest(),
                  position_rmse=float(np.sqrt(np.mean((prediction[:,:2]-vn[:,:2])**2))),velocity_rmse=float(np.sqrt(np.mean((prediction[:,2:]-vn[:,2:])**2))),
                  protocol='Same data as v2. Learned affine free/wall heads and residual bumper MLP in normal/tangent coordinates. Known contact geometry and coordinate transformations are supplied priors. Every response coefficient is fitted from observations, no simulator response or projection overrides inference. Training seed fixed before evaluation. Final confirmation seed 202609292 remains unused.')
    (ROOT/'assets/local-metadata.json').write_text(json.dumps(metadata,indent=2))
    # Independent PyTorch forward values in the local frame, transformed to world coordinates.
    mask=np.linalg.norm(vs[:,:2]+vs[:,2:]-.5,axis=1)<.115
    with torch.no_grad(): local=model(torch.tensor(vx,dtype=torch.float32)).numpy()+np.column_stack([vx,np.ones(len(vx))])@skip
    q=vs[mask,:2]+vs[mask,2:];normal=(q-.5)/np.linalg.norm(q-.5,axis=1,keepdims=True);tangent=np.column_stack([-normal[:,1],normal[:,0]])
    prediction[mask]=vs[mask]+.11*np.column_stack([local[:,:1]*normal+local[:,1:2]*tangent,local[:,2:3]*normal+local[:,3:4]*tangent])
    ids=np.r_[np.arange(32),np.flatnonzero(mask)[:32]]
    (ROOT/'assets/local-parity.json').write_text(json.dumps(dict(states=vs[ids].tolist(),nextStates=prediction[ids].tolist())))
    print(json.dumps({k:v for k,v in metadata.items() if k not in ['logs','protocol']},indent=2))
if __name__=='__main__':main()
