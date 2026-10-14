"""Specialized neural heads: supervised linear free head + nonlinear contact head."""
from pathlib import Path
import json,hashlib,time
import numpy as np
import torch
from train_improved import dataset,features,Network,export,ROOT

def main():
    seed=273;torch.set_num_threads(2);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    states,next_states=dataset(seed,180000);vs,vnext=dataset(seed+1,18000)
    x=torch.from_numpy(features(states));y=torch.from_numpy(((next_states-states)/.11).astype(np.float32))
    vx=torch.from_numpy(features(vs));vy=torch.from_numpy(((vnext-vs)/.11).astype(np.float32))
    contact=(x[:,4:9]<0).any(dim=1);vcontact=(vx[:,4:9]<0).any(dim=1)
    # A linear neural layer is sufficient in a smooth region. Its weights are fitted to observations, not assigned drag coefficients.
    free_x=torch.cat([x[~contact,:4].double(),torch.ones((int((~contact).sum()),1),dtype=torch.float64)],dim=1)
    free_weights=torch.linalg.lstsq(free_x,y[~contact].double()).solution
    model=Network();old=json.loads((ROOT/'host/experiments/dense-neural-v1/improved-model.json').read_text())
    with torch.no_grad():
        for i,layer in enumerate([model.layers[0],model.layers[2],model.layers[4]]):
            layer.weight.copy_(torch.tensor(old['weights'][i]).reshape(layer.in_features,layer.out_features).T);layer.bias.copy_(torch.tensor(old['biases'][i]))
        model.skip.weight.copy_(torch.tensor(old['skipWeight']).reshape(11,4).T);model.skip.bias.copy_(torch.tensor(old['skipBias']))
    cx=x[contact];cy=y[contact];opt=torch.optim.Adam(model.parameters(),lr=.0005)
    schedule=torch.optim.lr_scheduler.MultiStepLR(opt,[250,450],gamma=.3)
    logs=[];began=time.time()
    for epoch in range(600):
        order=torch.randperm(len(cx));total=0
        for ids in order.split(512):
            loss=(model(cx[ids])-cy[ids]).square().mean();opt.zero_grad();loss.backward();opt.step();total+=loss.item()*len(ids)
        schedule.step()
        if epoch%50==0 or epoch==599:
            with torch.no_grad():v=(model(vx[vcontact])-vy[vcontact]).square().mean().item()
            logs.append(dict(epoch=epoch+1,training_contact_mse=total/len(cx),validation_contact_mse=v))
            print(f'Contact epoch {epoch+1}/600 loss={total/len(cx):.6f} validation={v:.6f}',flush=True)
    path=ROOT/'assets/improved-model.json';export(model,path);m=json.loads(path.read_text());m['format']='gated-residual-mlp-v2';m['freeWeight']=free_weights[:4].numpy().ravel().tolist();m['freeBias']=free_weights[4].numpy().tolist();path.write_text(json.dumps(m,separators=(',',':')))
    with torch.no_grad():
        delta=np.column_stack([features(vs)[:,:4],np.ones(len(vs))])@free_weights.numpy()
        delta[vcontact.numpy()]=model(vx[vcontact]).numpy()
    prediction=vs+delta*.11
    meta=dict(seed=seed,validation_seed=seed+1,training_samples=len(states),contact_training_samples=int(contact.sum()),validation_samples=len(vs),contact_validation_samples=int(vcontact.sum()),epochs=600,architecture='geometry-routed linear neural free head and residual ReLU contact head [11,64,64,4]',seconds=time.time()-began,logs=logs,position_rmse=float(np.sqrt(np.mean((prediction[:,:2]-vnext[:,:2])**2))),velocity_rmse=float(np.sqrt(np.mean((prediction[:,2:]-vnext[:,2:])**2))),torch=torch.__version__,numpy=np.__version__,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),training_state_sha256=hashlib.sha256(states.tobytes()).hexdigest(),validation_state_sha256=hashlib.sha256(vs.tobytes()).hexdigest(),warm_start='dense-neural-v1 contact weights; preserved under host/experiments',protocol='Additional training observations spanning speed 0..0.115 and near boundaries. A geometric contact gate routes to learned neural outputs; no reflection/restitution equations run in inference. The free branch is a supervised linear neural layer fit with least squares; the contact branch is a trained residual MLP. This is a combined architecture/data/feature repair. New confirmation scenes remain unused until development selection is complete.')
    (ROOT/'assets/improved-metadata.json').write_text(json.dumps(meta,indent=2))
    ids=np.r_[np.flatnonzero(~vcontact.numpy())[:16],np.flatnonzero(vcontact.numpy())[:48]]
    (ROOT/'assets/improved-parity.json').write_text(json.dumps(dict(states=vs[ids].tolist(),nextStates=prediction[ids].tolist())))
    print(json.dumps({k:v for k,v in meta.items() if k not in ['logs','protocol']},indent=2))
if __name__=='__main__':main()
