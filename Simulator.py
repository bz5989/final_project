from Core import TaskInstance
from Environment import Environment

class Simulator:
    def __init__(self, env, agent, horizon):
        self.env = env
        self.agent = agent
        self.horizon = horizon

    def run(self):
        self.agent.reset()
        task_instances_available = []
        progress = {}
        results = []

        for step in self.env:
            timestep = step.t
            for task in step.task_instances:
                task_instances_available.append(task)
                progress[task.instance_id] = 0.0

            plan = self.agent.plan(list(task_instances_available), timestep)
            focus=plan[0]
            if focus: 
                progress[focus.instance_id] += 1.0
                if progress[focus.instance_id] >= focus.duration:
                    reward = focus.get_reward(timestep)
                    results.append((focus, timestep, reward))
                    task_instances_available.remove(focus) 

        return results
    
def summarise(agent_name, results, env):
    total_reward = sum(r for _, _, r in results)
    print(f"Agent {agent_name} completed {len(results)}/{len(env.all_tasks())} tasks with a total reward of {total_reward:.2f}")