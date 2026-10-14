"""Data -> inferred geometry -> contact-local neural model -> score.  No hand-given table constants.

1. Infer the table's geometry (drag, wall positions, bumper circle) from the ORIGINAL training transitions only.
2. Train the contact-local model (learned free / wall / bumper pieces, collisions in surface coordinates) using the
   INFERRED geometry to decide which piece handles each step.
3. Write a submission wrapper that routes with the inferred geometry at prediction time too.

Run from the puck-lab folder:
    python host/geometry_from_data_pipeline.py scratch/geofree expanded     # larger provided dataset (default)
    python host/geometry_from_data_pipeline.py scratch/geofree original     # strict: original 30,000 transitions only
Score it:
    python host/score_submission.py scratch/geofree/repair_local_inferred.py

Assumed (the FORM of the world, not its numbers): position moves by velocity, a drag law, reflecting straight walls, one round bumper.
Everything numeric (drag, wall positions, bumper centre and radius, and every response) comes from the data.
"""
import json,sys,time,pathlib
import numpy as np
import torch
from torch import nn
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=pathlib.Path(sys.argv[1] if len(sys.argv)>1 else ROOT/'scratch/geofree');OUT.mkdir(parents=True,exist_ok=True)
DATA=sys.argv[2] if len(sys.argv)>2 else 'expanded'
SCALE=.11                                   # speed normalisation for numerical conditioning, not geometry

# ---------------------------------------------------------------- 1. infer geometry from the original transitions
def infer_geometry():
    with np.load(ROOT/'assets/training-observations.npz',allow_pickle=False) as d:
        S,N=d['states'],d['next_states']
    v,nv=S[:,2:],N[:,2:];speed2=(v*v).sum(1);ok=speed2>1e-5
    drag=float(np.median(((nv*v).sum(1)[ok])/speed2[ok]))
    glide=np.column_stack([S[:,:2]+S[:,2:],drag*S[:,2:]])
    event=np.linalg.norm(N-glide,axis=1)>1e-4
    q=S[:,:2]+S[:,2:];moved=np.abs(N[:,:2]-q)>1e-4
    one_axis=event&(moved.sum(1)==1);two_axis=event&(moved.sum(1)==2)
    lo,hi=[],[]
    for k in (0,1):
        m=one_axis&moved[:,k];w=(q[m,k]+N[m,k])/2;up=S[m,2+k]>0
        hi+=list(w[up]);lo+=list(w[~up])
    wall_lo,wall_hi=float(np.median(lo)),float(np.median(hi))
    P=N[two_axis,:2];P=P[(P>wall_lo+.05).all(1)&(P<wall_hi-.05).all(1)]
    A=np.column_stack([2*P[:,0],2*P[:,1],np.ones(len(P))]);b=(P**2).sum(1)
    cx,cy,c=np.linalg.lstsq(A,b,rcond=None)[0]
    return dict(drag=drag,wall_lo=wall_lo,wall_hi=wall_hi,cx=float(cx),cy=float(cy),radius=float(np.sqrt(c+cx**2+cy**2)))

# ---------------------------------------------------------------- 2. contact-local training examples (geometry passed in)
def local_examples(s,next_s,g):
    lo,hi,cen,R=g['wall_lo'],g['wall_hi'],np.array([g['cx'],g['cy']]),g['radius']
    q=s[:,:2]+s[:,2:];delta=(next_s-s)/SCALE
    r=np.linalg.norm(q-cen,axis=1)
    free=((q>=lo)&(q<=hi)).all(1)&(r>=R)
    free_x=np.column_stack([s[free,2:]/SCALE,np.ones(free.sum())])
    wx=[];wy=[]
    for axis in range(2):
        for side,sign in [(lo,1),(hi,-1)]:
            clearance=(q[:,axis]-side)*sign;mask=clearance<0
            wx.append(np.column_stack([clearance[mask]/SCALE,s[mask,axis+2]*sign/SCALE,np.ones(mask.sum())]))
            wy.append(delta[mask][:,[axis,axis+2]]*sign)
    mask=r<R
    normal=(q[mask]-cen)/np.maximum(r[mask,None],1e-12);normal[r[mask]<1e-12]=[1,0]
    tangent=np.column_stack([-normal[:,1],normal[:,0]])
    x=np.column_stack([(r[mask]-R)/SCALE,np.sum(s[mask,2:]*normal,axis=1)/SCALE,np.sum(s[mask,2:]*tangent,axis=1)/SCALE])
    y=np.column_stack([np.sum(delta[mask,:2]*normal,axis=1),np.sum(delta[mask,:2]*tangent,axis=1),np.sum(delta[mask,2:]*normal,axis=1),np.sum(delta[mask,2:]*tangent,axis=1)])
    return free_x,delta[free],np.concatenate(wx),np.concatenate(wy),x,y

def load_data():
    if DATA=='expanded':
        with np.load(ROOT/'assets/neural-training.npz',allow_pickle=False) as d:
            return [d[k] for k in ['states','next_states','validation_states','validation_next_states']]
    with np.load(ROOT/'assets/training-observations.npz',allow_pickle=False) as d:
        S,N=d['states'],d['next_states']
    order=np.random.default_rng(42).permutation(1200)
    tr=(order[:960,None]*25+np.arange(25)).ravel();va=(order[960:,None]*25+np.arange(25)).ravel()
    return S[tr],N[tr],S[va],N[va]

def main():
    began=time.perf_counter()
    g=infer_geometry();(OUT/'geometry.json').write_text(json.dumps(g,indent=2))
    print('inferred geometry:',{k:round(v,4) for k,v in g.items()})
    seed=373;torch.set_num_threads(2);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    s,n,vs,vn=load_data()
    fx,fy,wx,wy,x,y=local_examples(s,n,g);*_,vx,vy=local_examples(vs,vn,g)
    print(f'data={DATA}: {len(s):,} steps, free {len(fx):,}, wall {len(wx):,}, bumper {len(x):,}')
    free=np.linalg.lstsq(fx,fy,rcond=None)[0];wall=np.linalg.lstsq(wx,wy,rcond=None)[0]
    skip=np.linalg.lstsq(np.column_stack([x,np.ones(len(x))]),y,rcond=None)[0]
    residual=y-np.column_stack([x,np.ones(len(x))])@skip
    target=torch.tensor(residual,dtype=torch.float32);tx=torch.tensor(x,dtype=torch.float32)
    model=nn.Sequential(nn.Linear(3,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,4))
    nn.init.zeros_(model[-1].weight);nn.init.zeros_(model[-1].bias)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,[500,800],gamma=.3)
    for epoch in range(1000):
        for ids in torch.randperm(len(x)).split(512):
            loss=(model(tx[ids])-target[ids]).square().mean();optimizer.zero_grad();loss.backward();optimizer.step()
        schedule.step()
        if epoch%250==0 or epoch==999:
            with torch.no_grad(): pred=model(torch.tensor(vx,dtype=torch.float32)).numpy()+np.column_stack([vx,np.ones(len(vx))])@skip
            print(f'  epoch {epoch+1}/1000: bumper validation MSE {float(np.mean((pred-vy)**2)):.9f}',flush=True)
    layers=[model[i] for i in [0,2,4]]
    checkpoint=dict(format='local-neural-v3',architecture=[3,32,32,4],activation='relu',outputScale=SCALE,geometry=g,
                    freeWeight=free.ravel().tolist(),wallWeight=wall.ravel().tolist(),skipWeight=skip.ravel().tolist(),
                    weights=[l.weight.detach().numpy().T.ravel().tolist() for l in layers],biases=[l.bias.detach().numpy().tolist() for l in layers])
    (OUT/'model.json').write_text(json.dumps(checkpoint,separators=(',',':'),allow_nan=False))
    (OUT/'repair_local_inferred.py').write_text(WRAPPER)
    print(f'wrote {OUT} in {time.perf_counter()-began:.0f} s')

WRAPPER='''"""Contact-local predictor that routes with geometry INFERRED from the data (model.json['geometry'])."""
import json,pathlib
import numpy as np
HERE=pathlib.Path(__file__).parent
SCALE=.11
class Repair:
    def __init__(self,neural,training_states,training_next_states):
        m=json.loads((HERE/'model.json').read_text());self.g=m['geometry']
        self.free=np.array(m['freeWeight']).reshape(3,4);self.wall=np.array(m['wallWeight']).reshape(3,2);self.skip=np.array(m['skipWeight']).reshape(4,4)
        dims=m['architecture'];self.weights=[np.array(w).reshape(dims[i],dims[i+1]) for i,w in enumerate(m['weights'])];self.biases=[np.array(b) for b in m['biases']]
    def __call__(self,states):
        g=self.g;lo,hi,cen,R=g['wall_lo'],g['wall_hi'],np.array([g['cx'],g['cy']]),g['radius']
        states=np.asarray(states,dtype=float);single=states.ndim==1;s=np.atleast_2d(states)
        q=s[:,:2]+s[:,2:]
        delta=np.column_stack([s[:,2:]/SCALE,np.ones(len(s))])@self.free
        for axis in range(2):
            for side,sign in [(lo,1),(hi,-1)]:
                clearance=(q[:,axis]-side)*sign;mask=clearance<0
                f=np.column_stack([clearance[mask]/SCALE,s[mask,axis+2]*sign/SCALE,np.ones(mask.sum())]);response=f@self.wall
                delta[mask,axis]=response[:,0]*sign;delta[mask,axis+2]=response[:,1]*sign
        offset=q-cen;r=np.linalg.norm(offset,axis=1);mask=r<R
        if mask.any():
            normal=offset[mask]/np.maximum(r[mask,None],1e-12);normal[r[mask]<1e-12]=[1,0]
            tangent=np.column_stack([-normal[:,1],normal[:,0]]);velocity=s[mask,2:]
            f=np.column_stack([(r[mask]-R)/SCALE,np.sum(velocity*normal,axis=1)/SCALE,np.sum(velocity*tangent,axis=1)/SCALE]);a=f
            for i,(w,b) in enumerate(zip(self.weights,self.biases)):
                a=a@w+b
                if i<2:a=np.maximum(a,0)
            response=a+np.column_stack([f,np.ones(len(f))])@self.skip
            delta[mask,:2]=response[:,:1]*normal+response[:,1:2]*tangent;delta[mask,2:]=response[:,2:3]*normal+response[:,3:4]*tangent
        result=s+delta*SCALE
        return result[0] if single else result
'''
if __name__=='__main__':main()
