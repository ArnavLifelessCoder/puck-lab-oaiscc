"""Editable neural training starter. No simulator calls or final-scene access."""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
import torch
from torch import nn
from neural_model import features, NeuralDynamics
ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', choices=['original','expanded'], default='original')
    parser.add_argument('--geometry', action='store_true')
    parser.add_argument('--free-head', action='store_true', help='Fit a separate affine neural head for free motion')
    parser.add_argument('--contact-weight', type=float, default=1)
    parser.add_argument('--width', type=int, default=48)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results/learned-repair.json')
    args = parser.parse_args()
    if args.epochs < 1 or args.width < 1 or not np.isfinite(args.contact_weight) or args.contact_weight <= 0:
        parser.error('epochs, width and contact-weight must be positive')
    torch.set_num_threads(2); torch.manual_seed(args.seed); torch.use_deterministic_algorithms(True)
    if args.data == 'expanded':
        with np.load(ROOT/'assets/neural-training.npz', allow_pickle=False) as data:
            train, target, validation, validation_target = [data[k] for k in ['states','next_states','validation_states','validation_next_states']]
        split = 'Independent state samples, training seed 273 and validation seed 274'
    else:
        with np.load(ROOT/'assets/training-observations.npz', allow_pickle=False) as data:
            states, next_states = data['states'], data['next_states']
        # The original export contains 1,200 contiguous 25-transition trajectories.
        assert states.shape == next_states.shape == (30000,4)
        order = np.random.default_rng(args.seed).permutation(1200)
        train_ids = (order[:960,None]*25+np.arange(25)).ravel()
        val_ids = (order[960:,None]*25+np.arange(25)).ravel()
        train,target,validation,validation_target = states[train_ids],next_states[train_ids],states[val_ids],next_states[val_ids]
        split = '960 training and 240 validation trajectories, no trajectory split across partitions'
    raw_x = features(train,args.geometry); raw_y = target-train
    xm,xs = raw_x.mean(0),np.maximum(raw_x.std(0),1e-8)
    ym,ys = raw_y.mean(0),np.maximum(raw_y.std(0),1e-8)
    x = torch.tensor((raw_x-xm)/xs,dtype=torch.float32); y = torch.tensor((raw_y-ym)/ys,dtype=torch.float32)
    vx = torch.tensor((features(validation,args.geometry)-xm)/xs,dtype=torch.float32)
    q = train[:,:2]+train[:,2:]
    contact = ((q<.02)|(q>.98)).any(1)|(np.linalg.norm(q-.5,axis=1)<.115)
    weights = torch.tensor(np.where(contact,args.contact_weight,1),dtype=torch.float32)
    model = nn.Sequential(nn.Linear(x.shape[1],args.width),nn.ReLU(),nn.Linear(args.width,args.width),nn.ReLU(),nn.Linear(args.width,4))
    optimizer = torch.optim.Adam(model.parameters(),lr=.001)
    began=time.perf_counter()
    for epoch in range(args.epochs):
        for ids in torch.randperm(len(x)).split(1024):
            loss = ((model(x[ids])-y[ids]).square().mean(1)*weights[ids]).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    layers = [model[i] for i in [0,2,4]]
    checkpoint = dict(format='team-mlp-v1',geometry=args.geometry,architecture=[x.shape[1],args.width,args.width,4],
                      weights=[l.weight.detach().numpy().T.ravel().tolist() for l in layers],biases=[l.bias.detach().numpy().tolist() for l in layers],
                      inputMean=xm.tolist(),inputScale=xs.tolist(),outputMean=ym.tolist(),outputScale=ys.tolist())
    if args.free_head:
        free_inputs = np.column_stack([train[~contact,2:]/.11, np.ones((~contact).sum())])
        free_targets = (target[~contact]-train[~contact])/.11
        checkpoint['freeWeight'] = np.linalg.lstsq(free_inputs,free_targets,rcond=None)[0].ravel().tolist()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(checkpoint,allow_nan=False))
    with torch.no_grad(): prediction=validation+model(vx).numpy()*ys+ym
    if args.free_head:
        qv=validation[:,:2]+validation[:,2:]
        free=((qv>=.02)&(qv<=.98)).all(1)&(np.linalg.norm(qv-.5,axis=1)>=.115)
        prediction[free]=validation[free]+.11*np.column_stack([validation[free,2:]/.11,np.ones(free.sum())])@np.array(checkpoint['freeWeight']).reshape(3,4)
    portable=NeuralDynamics(args.output)(validation[:64])
    if not np.allclose(portable,prediction[:64],atol=2e-6,rtol=0): raise ValueError('Export parity failed')
    report=dict(configuration={**vars(args),'output':str(args.output)},split=split,training_samples=len(train),validation_samples=len(validation),
                training_sha256=hashlib.sha256(train.tobytes()).hexdigest(),validation_sha256=hashlib.sha256(validation.tobytes()).hexdigest(),
                position_rmse=float(np.sqrt(np.mean((prediction[:,:2]-validation_target[:,:2])**2))),
                velocity_rmse=float(np.sqrt(np.mean((prediction[:,2:]-validation_target[:,2:])**2))),
                seconds=time.perf_counter()-began,torch=torch.__version__,numpy=np.__version__)
    args.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    print('This is your candidate network, not a guaranteed improvement. Run check_submission.py and report all outcomes.')

if __name__ == '__main__': main()
