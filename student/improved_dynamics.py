"""Inference for the trained residual MLP. No simulator or calibrated response imports."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def features(states):
    states=np.atleast_2d(np.asarray(states,dtype=float))
    q=states[:,:2]+states[:,2:];d=q-.5;r=np.linalg.norm(d,axis=1,keepdims=True)
    normal=d/np.maximum(r,1e-8)
    return np.column_stack([(states[:,:2]-.5)*2,states[:,2:]/.11,(q[:,0]-.02)/.11,(.98-q[:,0])/.11,(q[:,1]-.02)/.11,(.98-q[:,1])/.11,(r[:,0]-.115)/.11,normal])

class ImprovedDynamics:
    def __init__(self):
        m=json.loads((ROOT/'assets/improved-model.json').read_text())
        if m['format'] not in ['residual-geometry-mlp-v1','gated-residual-mlp-v2']:raise ValueError('Unexpected checkpoint format')
        dims=m['architecture'];self.weights=[np.array(w).reshape(dims[i],dims[i+1]) for i,w in enumerate(m['weights'])];self.biases=[np.array(b) for b in m['biases']]
        self.skip_weight=np.array(m['skipWeight']).reshape(11,4);self.skip_bias=np.array(m['skipBias']);self.scale=m['outputScale'];self.gated=m['format']=='gated-residual-mlp-v2'
        if self.gated:self.free_weight=np.array(m['freeWeight']).reshape(4,4);self.free_bias=np.array(m['freeBias'])
    def __call__(self,states):
        states=np.asarray(states,dtype=float);single=states.ndim==1;x=features(states);a=x
        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            a=a@w+b
            if i<2:a=np.maximum(a,0)
        delta=a+x@self.skip_weight+self.skip_bias
        if self.gated:
            contact=(x[:,4:9]<0).any(axis=1)
            delta[~contact]=x[~contact,:4]@self.free_weight+self.free_bias
        output=np.atleast_2d(states)+delta*self.scale
        return output[0] if single else output
