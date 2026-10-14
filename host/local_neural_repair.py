from local_dynamics import LocalDynamics
class Repair:
    def __init__(self,neural,training_states,training_next_states):self.model=LocalDynamics()
    def __call__(self,states):return self.model(states)
