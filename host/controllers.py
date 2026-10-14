"""Controllers with explicit information budgets and a common candidate pool."""
import json
from pathlib import Path
import numpy as np
from lab import Dynamics, candidates
ROOT=Path(__file__).resolve().parents[1]

def contacts(states):
    states=np.atleast_2d(states)
    positions=states[:,:2]+states[:,2:]
    walls=(positions<.02)|(positions>.98)
    positions=np.where(positions<.02,.04-positions,np.where(positions>.98,1.96-positions,positions))
    delta=positions-.5
    distance=np.linalg.norm(delta,axis=1)
    bumper=distance<.115
    normal=delta/np.maximum(distance[:,None],1e-12)
    normal[distance==0]=[1,0]
    return positions,walls,bumper,normal

def structured_step(states,parameters,neural=None):
    """Known kinematics/contact geometry; response coefficients learned from observations."""
    s=np.atleast_2d(np.asarray(states,dtype=float))
    p,walls,bumper,n=contacts(s)
    velocity=s[:,2:]*parameters['drag']
    velocity=np.where(walls,-velocity*parameters['wall_restitution'],velocity)
    inward=np.sum(velocity*n,axis=1)
    active=bumper&(inward<0)
    velocity[active]-=(1+parameters['bumper_restitution'])*inward[active,None]*n[active]
    p[bumper]=.5+n[bumper]*.115
    if neural is not None:
        # Ablation: keep neural free-motion velocity, repair contact and position integration only.
        free=~(np.any(walls,axis=1)|bumper)
        velocity[free]=neural(s[free])[:,2:]
    return np.column_stack([p,velocity])

def fit_parameters(states,next_states):
    _,walls,bumper,n=contacts(states)
    free=~(np.any(walls,axis=1)|bumper)
    incoming=states[free,2:].ravel();outgoing=next_states[free,2:].ravel()
    drag=float(incoming@outgoing/(incoming@incoming))
    wall_x=(states[:,2:]*drag)[walls];wall_y=next_states[:,2:][walls]
    wall=float(-(wall_x@wall_y)/(wall_x@wall_x))
    incoming=np.sum(states[:,2:]*drag*n,axis=1)
    valid=bumper&(incoming<0)&~np.any(walls,axis=1)
    outgoing=np.sum(next_states[:,2:]*n,axis=1)
    restitution=float(-(incoming[valid]@outgoing[valid])/(incoming[valid]@incoming[valid]))
    values=dict(drag=drag,wall_restitution=wall,bumper_restitution=restitution)
    if not all(np.isfinite(v) and 0<=v<=1 for v in values.values()):raise ValueError('Unphysical fitted coefficient.')
    return values,dict(free=int(free.sum()),wall_components=int(walls.sum()),bumper=int(valid.sum()))

def batch_rollout(start,actions,transition,steps=35):
    states=np.column_stack([np.tile(start,(len(actions),1)),actions]);paths=[states]
    for _ in range(steps):
        states=transition(states)
        if not np.isfinite(states).all():raise ValueError('Nonfinite rollout.')
        paths.append(states)
    return np.stack(paths,axis=1)

def costs(paths,target):
    miss=np.linalg.norm(paths[:,-1,:2]-target,axis=1)
    outside=np.any((paths[:,:,:2]<.02)|(paths[:,:,:2]>.98),axis=2).sum(axis=1)
    collision=(np.linalg.norm(paths[:,:,:2]-.5,axis=2)<.115-1e-7).sum(axis=1)
    return miss+.025*(outside+collision)

def transitions(model,parameters):
    return {'neural':model,'contact-repair':lambda s:structured_step(s,parameters,model),'calibrated':lambda s:structured_step(s,parameters),'oracle':lambda s:structured_step(s,dict(drag=.985,wall_restitution=.85,bumper_restitution=.85))}

def scene_set(seed,count):
    rng=np.random.default_rng(seed);scenes=[]
    def point():
        while True:
            p=rng.uniform(.08,.92,2)
            if np.linalg.norm(p-.5)>.18:return p
    while len(scenes)<count:
        start,target=point(),point()
        if np.linalg.norm(start-target)>=.30:scenes.append(dict(start=start.tolist(),target=target.tolist()))
    return scenes
