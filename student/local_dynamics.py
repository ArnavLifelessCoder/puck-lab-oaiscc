"""Learned dynamics in contact-local coordinates. No simulator response equations."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

class LocalDynamics:
    def __init__(self,path=None):
        self.model=json.loads(Path(path or ROOT/'assets/local-model.json').read_text())
        m=self.model
        if m['format']!='local-neural-v3':raise ValueError('Unexpected local neural format')
        self.free=np.array(m['freeWeight']).reshape(3,4)
        self.wall=np.array(m['wallWeight']).reshape(3,2)
        self.skip=np.array(m['skipWeight']).reshape(4,4)
        dims=m['architecture']
        self.weights=[np.array(w).reshape(dims[i],dims[i+1]) for i,w in enumerate(m['weights'])]
        self.biases=[np.array(b) for b in m['biases']]
    def __call__(self,states):
        states=np.asarray(states,dtype=float);single=states.ndim==1;s=np.atleast_2d(states)
        q=s[:,:2]+s[:,2:];scale=.11
        delta=np.column_stack([s[:,2:]/scale,np.ones(len(s))])@self.free
        for axis in range(2):
            for side,sign in [(.02,1),(.98,-1)]:
                clearance=(q[:,axis]-side)*sign;mask=clearance<0
                f=np.column_stack([clearance[mask]/scale,s[mask,axis+2]*sign/scale,np.ones(mask.sum())])
                response=f@self.wall
                delta[mask,axis]=response[:,0]*sign;delta[mask,axis+2]=response[:,1]*sign
        offset=q-.5;r=np.linalg.norm(offset,axis=1);mask=r<.115
        if mask.any():
            normal=offset[mask]/np.maximum(r[mask,None],1e-12)
            normal[r[mask]<1e-12]=[1,0]
            tangent=np.column_stack([-normal[:,1],normal[:,0]])
            velocity=s[mask,2:]
            f=np.column_stack([(r[mask]-.115)/scale,np.sum(velocity*normal,axis=1)/scale,np.sum(velocity*tangent,axis=1)/scale])
            a=f
            for i,(w,b) in enumerate(zip(self.weights,self.biases)):
                a=a@w+b
                if i<2:a=np.maximum(a,0)
            response=a+np.column_stack([f,np.ones(len(f))])@self.skip
            delta[mask,:2]=response[:,:1]*normal+response[:,1:2]*tangent
            delta[mask,2:]=response[:,2:3]*normal+response[:,3:4]*tangent
        result=s+delta*scale
        return result[0] if single else result
