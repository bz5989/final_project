import numpy

class Task():
    def __init__(self, type: str, type_params: dict):
        self.type = type
        self.type_params = type_params
        # how long the task takes
        self.length = type_params['length_distribution'].sample()
        
        # due date of task
        self.due_date = type_params['due_date_distribution'].sample()
        self.total_value = type_params['total_value']
        # function
        self.late_penalty = type_params['late_penalty']

        # modifiable by agent
        self.progress = 0
    
    # returns if task is done after progress this step
    def progress(self):
        self.progress += 1
        if self.progress >= self.length:
            return True
        return False

class Agent():
    def __init__(self, name: str):
        self.name = name
        self.reward = 0
    
    def receive_reward(self, reward: int):
        self.reward += reward

class Scheduler_Agent():
    def __init__(self, horizon: int, types: dict, agent: Agent):
        self.horizon = horizon
        self.types = types
        self.schedule_window = self.initialize_schedule()
        self.history = {}
        self.agent = agent

    def initialize_schedule(self):
        return [None] * self.horizon

    def receive_task(self, task: Task):
        self.insert_task(task)

    def insert_task_first_possible(self, task: Task):
        # insert task into 
        put_task = False
        for i in self.horizon:
            if self.check_free(i, i+task.length):
                self.put_task(i, i+task.length, task)
                put_task = True
                break
        return put_task
    
    def check_free(self, start: int, end: int):
        for i in range(start, end):
            if self.schedule_window[i] != None:
                return False
        return True
    
    def put_task(self, start: int, end: int, task: Task):
        for i in range(start, end):
            self.schedule_window[i] = Task
        return True
    
    def step(self):
        if self.schedule_window[0] != None:
            task = self.schedule_window[0]
            complete = task.progress()
            if complete:
                self.mark_complete(task)
        self.shift_window()
    
    def shift_window(self):
        print("Shift window forward one step")
    
    def receive_reward(self, reward: int):
        self.agent.receive_reward(reward)
    
class environment():
    def __init__(self, types):
        self.types = types
    
    def publish_task(self, task):
        return task
    
    def check_tasks(self):
        for type in self.types:
            if type.sample() < type.valid:
                self.publish_task()
# order of events for each time slot (assume)
# end of current time chunk
# Agent completes tasks -> environment
# Environment marks late tasks -> agent
# start of next time chunk
# Environment sends tasks -> agent