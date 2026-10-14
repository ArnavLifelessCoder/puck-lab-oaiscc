"""Editable team API. Learned planning and simulator evaluation stay separate."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

class Dynamics:
    def __init__(self):
        model = json.loads((ROOT / 'assets/model.json').read_text())
        dims = model['architecture']
        self.weights = [np.array(w).reshape(dims[i], dims[i+1]) for i, w in enumerate(model['weights'])]
        self.biases = [np.array(b) for b in model['biases']]
        self.input_mean = np.array(model['inputMean'])
        self.input_scale = np.array(model['inputScale'])
        self.output_mean = np.array(model['outputMean'])
        self.output_scale = np.array(model['outputScale'])

    def __call__(self, state):
        state = np.asarray(state, dtype=float)
        hidden = (state - self.input_mean) / self.input_scale
        for i, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            hidden = hidden @ weights + bias
            if i < 2:
                hidden = np.tanh(hidden)
        return state + hidden * self.output_scale + self.output_mean

def physics(state):
    s = np.asarray(state, dtype=float).copy()
    s[:2] += s[2:]
    s[2:] *= .985
    for i in range(2):
        if s[i] < .02:
            s[i] = .04 - s[i]
            s[i+2] = abs(s[i+2]) * .85
        elif s[i] > .98:
            s[i] = 1.96 - s[i]
            s[i+2] = -abs(s[i+2]) * .85
    delta = s[:2] - .5
    d = np.linalg.norm(delta)
    if d < .115:
        normal = delta / d if d else np.array([1., 0.])
        s[:2] = .5 + normal * .115
        inward = s[2:] @ normal
        if inward < 0:
            s[2:] -= 1.85 * inward * normal
    return s

def rollout(start, action, steps, transition):
    states = [np.r_[start, action]]
    for _ in range(steps):
        states.append(transition(states[-1]))
    return np.array(states)

def candidates(count, max_speed=.11, seed=123):
    """Same nested action pool as the browser, including its RNG sequence."""
    state = seed
    def random():
        nonlocal state
        state = (1664525 * state + 1013904223) & 0xffffffff
        return state / 4294967296
    actions = []
    for _ in range(count):
        angle = random() * 2 * np.pi
        speed = .008 + random() * (max_speed - .008)
        actions.append([np.cos(angle) * speed, np.sin(angle) * speed])
    return np.array(actions)

def predicted_paths(model, start, actions, steps=35):
    states = np.column_stack([np.tile(start, (len(actions), 1)), actions])
    paths = [states]
    for _ in range(steps):
        states = model(states)
        paths.append(states)
    return np.stack(paths, axis=1)

def choose(model, start, target, budget=512, max_speed=.11, steps=35, speed_penalty=0):
    actions = candidates(budget, max_speed)
    paths = predicted_paths(model, start, actions, steps)
    end_error = np.linalg.norm(paths[:, -1, :2] - target, axis=1)
    outside = np.any((paths[:, :, :2] < .02) | (paths[:, :, :2] > .98), axis=2).sum(axis=1)
    obstacle = (np.linalg.norm(paths[:, :, :2] - .5, axis=2) < .115-1e-7).sum(axis=1)
    # Keep this scoring rule fixed for neural-repair comparisons. Physics here would leak the answer.
    costs = end_error + .025 * outside + .025 * obstacle + speed_penalty * np.linalg.norm(actions, axis=1)
    selected = int(np.argmin(costs))
    return actions[selected], paths[selected], selected

def experiment(model, start=(.16,.7), target=(.83,.72), **config):
    action, predicted, selected = choose(model, start, target, **config)
    actual = rollout(start, action, config.get('steps', 35), physics)
    return dict(action=action, selected=selected, predicted=predicted, actual=actual,
                predicted_miss=float(np.linalg.norm(predicted[-1,:2]-target)),
                actual_miss=float(np.linalg.norm(actual[-1,:2]-target)))

def plot_shot(result, start=(.16,.7), target=(.83,.72)):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(6,6))
    ax.add_patch(Circle((.5,.5),.115,color='#dce1e8'))
    ax.add_patch(Circle(target,.065,fill=False,color='#28764c',lw=2))
    for key, color, style in [('predicted','#727987','--'),('actual','#245bd6','-')]:
        p = result[key]
        ax.plot(p[:,0],p[:,1],style,color=color,label=key)
        ax.scatter(*p[-1,:2],color=color,s=35)
    ax.scatter(*start,color='black',s=35)
    ax.set(xlim=(0,1),ylim=(1,0),aspect='equal',xlabel='x',ylabel='y')
    ax.legend();fig.tight_layout()
    return fig
