"""Host reveal: the frozen trained neural model, without simulator access."""
from improved_dynamics import ImprovedDynamics
class Repair:
    def __init__(self, neural, training_states, training_next_states):
        self.model=ImprovedDynamics()
    def __call__(self, states):
        return self.model(states)
