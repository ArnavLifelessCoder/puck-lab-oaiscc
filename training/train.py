"""Train a next-state network on independent simulated puck trajectories."""
from pathlib import Path
import hashlib,json,platform,time
import numpy as np
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
SEED=71
OBSTACLE=np.array([.5,.5])
RADIUS=.115

def step(s):
    p=s[:2]+s[2:]
    v=s[2:]*.985
    for j in range(2):
        if p[j]<.02:
            p[j]=.04-p[j];v[j]=abs(v[j])*.85
        elif p[j]>.98:
            p[j]=1.96-p[j];v[j]=-abs(v[j])*.85
    delta=p-OBSTACLE
    distance=np.linalg.norm(delta)
    if distance<RADIUS:
        normal=delta/max(distance,1e-12) if distance else np.array([1.,0.])
        p=OBSTACLE+normal*RADIUS
        dot=v@normal
        if dot<0:v=v-1.85*dot*normal
    return np.r_[p,v]

def generate(rng,count):
    states=[];targets=[];groups=[]
    for g in range(count):
        p=rng.uniform(.08,.92,2)
        while np.linalg.norm(p-OBSTACLE)<RADIUS+.02:p=rng.uniform(.08,.92,2)
        angle=rng.uniform(-np.pi,np.pi);speed=rng.uniform(.008,.04)
        s=np.r_[p,speed*np.array([np.cos(angle),np.sin(angle)])]
        for t in range(25):
            next_s=step(s);states.append(s);targets.append(next_s-s);groups.append(g);s=next_s
    return np.array(states),np.array(targets)

def save(name,value):
    (ROOT/'assets'/name).write_text(json.dumps(value,separators=(',',':'),allow_nan=False),encoding='utf-8')

def main():
    start=time.time()
    x,y=generate(np.random.default_rng(SEED),1200)
    vx,vy=generate(np.random.default_rng(SEED+1),160)
    sx=StandardScaler().fit(x);sy=StandardScaler().fit(y)
    model=MLPRegressor(hidden_layer_sizes=(48,48),activation='tanh',solver='adam',batch_size=256,learning_rate_init=.001,random_state=SEED,max_iter=1)
    tx=sx.transform(x);ty=sy.transform(y)
    losses=[]
    with threadpool_limits(limits=2):
        for epoch in range(65):
            model.partial_fit(tx,ty);losses.append(float(model.loss_))
            if epoch%10==0:print(f'Epoch {epoch+1}/65, loss {model.loss_:.5f}',flush=True)
        pred=sy.inverse_transform(model.predict(sx.transform(vx)))
    output=dict(architecture=[4,48,48,4],activation='tanh',weights=[w.ravel().tolist() for w in model.coefs_],biases=[b.tolist() for b in model.intercepts_],inputMean=sx.mean_.tolist(),inputScale=sx.scale_.tolist(),outputMean=sy.mean_.tolist(),outputScale=sy.scale_.tolist())
    save('model.json',output)
    save('parity.json',dict(states=vx[:12].tolist(),nextStates=(vx[:12]+pred[:12]).tolist(),physicsNextStates=[step(s).tolist() for s in vx[:12]]))
    meta=dict(seed=SEED,validationSeed=SEED+1,trainingTrajectories=1200,validationTrajectories=160,stepsPerTrajectory=25,trainingSamples=len(x),validationSamples=len(vx),trainingSpeed=[.008,.04],epochs=65,losses=losses,positionRMSE=float(np.sqrt(np.mean((pred[:,:2]-vy[:,:2])**2))),velocityRMSE=float(np.sqrt(np.mean((pred[:,2:]-vy[:,2:])**2))),seconds=time.time()-start,numpy=np.__version__,sklearn=sklearn.__version__,python=platform.python_version(),modelSHA256=hashlib.sha256((ROOT/'assets/model.json').read_bytes()).hexdigest(),protocol='Fixed seed and 65 epochs. Independent trajectory sets with different random seeds; scalers fit only on training. Validation is ordinary low-speed motion, not a control-performance test. Physics is synthetic and deterministic. No test or scene-based checkpoint selection.')
    save('metadata.json',meta)
    print(json.dumps({k:v for k,v in meta.items() if k!='losses'},indent=2))
if __name__=='__main__':main()
