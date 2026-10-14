"""Portable inference for a team-trained neural network, without PyTorch at runtime."""
from pathlib import Path
import json
import numpy as np

def features(states, geometry=False):
    states = np.asarray(states, dtype=float)
    if states.ndim not in (1, 2) or states.shape[-1] != 4 or not np.isfinite(states).all():
        raise ValueError("Expected finite states with shape (4,) or (N, 4)")
    if not geometry:
        return states
    q = states[..., :2] + states[..., 2:]
    delta = q - .5
    radius = np.linalg.norm(delta, axis=-1, keepdims=True)
    normal = delta / np.maximum(radius, 1e-8)
    return np.concatenate([states, q[..., :1]-.02, .98-q[..., :1], q[..., 1:2]-.02,
                           .98-q[..., 1:2], radius-.115, normal], axis=-1)

class NeuralDynamics:
    def __init__(self, path):
        model = json.loads(Path(path).read_text())
        if model['format'] != 'team-mlp-v1':
            raise ValueError('Unknown team checkpoint format')
        self.geometry = model['geometry']
        self.free_weight = np.array(model['freeWeight']).reshape(3,4) if 'freeWeight' in model else None
        if self.free_weight is not None and not np.isfinite(self.free_weight).all(): raise ValueError('Invalid free-motion weights')
        dims = model['architecture']
        if len(dims) != 4 or dims[0] != (11 if self.geometry else 4) or dims[-1] != 4:
            raise ValueError('Invalid network architecture')
        self.weights = [np.array(w).reshape(dims[i], dims[i+1]) for i,w in enumerate(model['weights'])]
        self.biases = [np.array(b) for b in model['biases']]
        self.input_mean = np.array(model['inputMean']); self.input_scale = np.array(model['inputScale'])
        self.output_mean = np.array(model['outputMean']); self.output_scale = np.array(model['outputScale'])
        for value in [*self.weights, *self.biases, self.input_mean, self.input_scale, self.output_mean, self.output_scale]:
            if not np.isfinite(value).all(): raise ValueError('Non-finite checkpoint')
        if np.any(self.input_scale <= 0) or np.any(self.output_scale <= 0): raise ValueError('Invalid scales')

    def __call__(self, states):
        states = np.asarray(states, dtype=float)
        hidden = (features(states, self.geometry)-self.input_mean)/self.input_scale
        for i,(weight,bias) in enumerate(zip(self.weights,self.biases)):
            hidden = hidden @ weight + bias
            if i < 2: hidden = np.maximum(hidden, 0)
        result = states + hidden*self.output_scale+self.output_mean
        if self.free_weight is not None:
            single=states.ndim==1; batch=np.atleast_2d(states); output=np.atleast_2d(result)
            q=batch[:,:2]+batch[:,2:]
            free=((q>=.02)&(q<=.98)).all(1)&(np.linalg.norm(q-.5,axis=1)>=.115)
            output[free]=batch[free]+.11*np.column_stack([batch[free,2:]/.11,np.ones(free.sum())])@self.free_weight
            result=output[0] if single else output
        return result
