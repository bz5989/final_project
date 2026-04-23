
import numpy as np

class TaskInstance:
    def __init__(self, task_category, instance_id, start_time, end_time, reward, penalty_fn):
        self.task_category = task_category
        self.instance_id = instance_id

        self.start_time = start_time
        self.end_time = end_time
        self.reward = reward
        self.penalty_fn = penalty_fn

    def __repr__(self):
        return f"[Task Instance | Reward {self.reward} | Timing  {self.start_time} -> {self.end_time}]"
    
    def get_reward(self, current_time):
        overdue_time = self.end_time - current_time
        penalty = self.penalty_fn(overdue_time)
        return self.reward - penalty
    
    def get_duration(self): return self.end_time - self.start_time
    def get_start_time(self): return self.start_time
    def get_end_time(self): return self.end_time
    def get_label(self): return self.task_category.name

class TaskCategory:
    def __init__(self, name, category_seed, mean_time, std_time, mean_reward, std_reward, penalty_fn = None):
        self.name = name
        self.category_seed = category_seed
        self.instance_count = 0

        self.mean_time = mean_time
        self.std_time = std_time
        self.mean_reward = mean_reward
        self.std_reward = std_reward
        self.penalty_fn = penalty_fn or (lambda overdue_time: max(0,overdue_time))

    def _get_instance_seed(self, instance_id):
        ss = np.random.SeedSequence([self.category_seed, instance_id])
        return ss.generate_state(1)[0]
    
    def create_instance(self, start_time=0, instance_id=None):
        if not instance_id: 
            instance_id = self.instance_count
            self.instance_count += 1
        
        seed = self._get_instance_seed(instance_id)
        subseed = np.random.default_rng(seed)
        
        task_start_time = start_time
        task_end_time = start_time + max(1,subseed.normal(self.mean_time, self.std_time))
        task_reward = max(0,subseed.normal(self.mean_reward, self.std_reward))

        return TaskInstance(
            task_category=self,
            instance_id=instance_id,
            start_time = task_start_time,
            end_time = task_end_time,
            reward = task_reward,
            penalty_fn=self.penalty_fn
        )

class TaskGenerator():
    def __init__(self, task_category, generator_seed, probability):
        self.task_category = task_category
        self.generator_seed = generator_seed
        self.probability = probability
        self.reset()
    
    def reset(self):
        self.instance_count = 0

    def _get_instance_seed(self, instance_id):
        ss = np.random.SeedSequence([self.generator_seed, instance_id])
        return ss.generate_state(1)[0]
    
    def create_instance(self, start_time=0, instance_id=None):
        if not instance_id:
            instance_id = self.instance_count
            self.instance_count += 1

        seed = self._get_instance_seed(instance_id)
        subseed = np.random.default_rng(seed)

        if subseed.uniform(0.0,1.0) > self.probability: return None
        return self.task_category.create_instance(start_time)

class Environment:
    def __init__(self, seed=42, generators=[], timesteps=100):       
        self.generators = generators
        self.timesteps = timesteps
        self.cum_history = []
        self.history = []

        for generator in self.generators: generator.reset()

        for timestep in range(timesteps):
            timestep_history = []
            for generator in self.generators:
                task_instance = generator.create_instance(timestep)
                timestep_history.append(task_instance)
            self.history.append(timestep_history)

            if not self.cum_history: self.cum_history = [x for x in self.history]
            else: self.cum_history = [x for x in self.cum_history[-1]] + [x for x in self.history]

    def get_history(self, timestep):
        if timestep >= self.timesteps or timestep < 0: raise Exception("Timestep out of history")
        return self.history[timestep]
    
    def get_cum_history(self, timestep):
        if timestep >= self.timesteps or timestep < 0: raise Exception("Timestep out of history")
        return self.cum_history[timestep]

# Then add stuff so that its statically readable, then the rest should be down to the Agent, the Environment should not change, 
# We can query the environment and such, but ultimately the environment should not change
# The Agent for now does constant work with the environment but we can also seed that
# Ultimately the Agent needs to keep track of these things, so build a simple random agent after this
# Extend TaskGenerator so there is independent, dependent, poisson, etc cooldown etc.
# Streamline this a littlebit, with all the generation see if we can remove the need for so much?
        
if __name__ == "__main__":
    print("Running from Environment.py")

    cat = TaskCategory("Type 1", 42, 1, 0.1, 1, 0.1)
    gen = TaskGenerator(cat, 42, 0.1)

    for x in range(100):
        inst = gen.create_instance(0)
        if inst: print(inst)
        else: print("No instance")