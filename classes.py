import torch
from torch.distributions import Normal

class Schedule():
    def __init__(self, name, types):
        self.name = name
        self.types = types
        self.assignments = {type.label: [] for type in types}
        self.history = []
    
    def add_assignment(self, assignment):
        self.assignments[assignment.type.label].append(assignment)
    
    def complete_assignment(self, assignment):
        self.assignments[assignment.type.label].remove(assignment)
        self.history.append(assignment)

class Assignment():
    def __init__(self, name, type, due_date, value):
        self.name = name
        self.type = type
        # either time of expiry or remaining time needed
        self.due_date = due_date
        self.value = value

        self.true_time = None
        self.true_progress = self.type.true_progress_distribution.sample()

        # guess of values
        self.guess_time = None
        self.guess_progress = None

        self.current_time = None
        self.current_progress = None

    def guess_remaining_time(self):
        return self.guess_time - self.current_time
    
    def guess_remaining_progress(self):
        return self.guess_progress - self.current_progress
    
    def step_time(self):
        self.type.decay()
    
    def step_progress(self, progress_increment):
        self.current_progress += progress_increment
        self.current_time += self.type.time_equivalent(progress_increment)

class Type():
    def __init__(self, label):
        self.label = label
        # number of progress steps
        self.true_progress_distribution = Normal(mean=10, std=2)
        self.flow_state_distribution = Normal(mean=0.5, std=0.1)
        self.drop_off_rate = None
        self.current_flow = None
        self.estimated_time = None
        self.estimated_progress = None
    
    def time_equivalent(self, progress_increment):
        return progress_increment * self.estimated_time / self.estimated_progress
    
    def decay(self):
        self.current_flow -= self.drop_off_rate
