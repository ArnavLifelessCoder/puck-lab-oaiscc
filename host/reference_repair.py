"""Instructor solution, deliberately uses an explicit physics prior."""
from controllers import fit_parameters,structured_step
class Repair:
    def __init__(self,neural,training_states,training_next_states):
        self.parameters,self.counts=fit_parameters(training_states,training_next_states)
    def __call__(self,states):
        return structured_step(states,self.parameters)
