import numpy as np
from Core import TaskInstance

class RandomAgent:
    name="Random"
    def __init__(self, horizon, seed=0):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.horizon = horizon

    def plan(self, task_instances, current_time):
        tasks = list(task_instances)
        self._rng.shuffle(tasks)
        return (tasks + [None] * self.horizon)[:self.horizon]
    
    def reset(self):
        self._rng = np.random.default_rng(self.seed)

class GreedyAgent:
    name="Greedy"
    def __init__(self, horizon):
        self.horizon = horizon

    def plan(self, tasks_instances, current_time):
        ranked = sorted(tasks_instances, key=lambda t:t.get_reward(current_time), reverse=True)
        return (ranked+[None]*self.horizon)[:self.horizon]
    
    def reset(self):
        return