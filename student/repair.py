"""Team submission: use your trained checkpoint, or explicitly reproduce the baseline."""
from pathlib import Path
import numpy as np
from neural_model import NeuralDynamics

class Repair:
    def __init__(self, neural, training_states, training_next_states):
        checkpoint = Path(__file__).parent/'results/learned-repair.json'
        self.neural = NeuralDynamics(checkpoint) if checkpoint.exists() else neural
        self.training_samples = len(training_states)
        print('Using team-trained neural checkpoint.' if checkpoint.exists() else 'No team checkpoint: using the unchanged original neural network.')

    def __call__(self, states):
        # Replace or extend neural inference here. Simulator response equations are not a repair.
        return self.neural(np.asarray(states, dtype=float))
