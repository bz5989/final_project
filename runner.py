from Core import TaskCategory, TaskGenerator, linear_penalty, no_penalty
from Environment import Environment
from Agents import RandomAgent, GreedyAgent
from Simulator import Simulator, summarise

quick_tasks = TaskCategory(
    name="Quick",
    category_seed=1,
    mean_time=1,
    std_time=0.3,
    mean_buffer_time=1.0,
    std_buffer_time=0.4,
    mean_reward=3.0,
    std_reward=0.5,
    penalty_fn=linear_penalty
)

big_tasks = TaskCategory(
    name="Big",
    category_seed=2,
    mean_time=5,
    std_time=1.0,
    mean_buffer_time=3.0,
    std_buffer_time=1.0,
    mean_reward=10.0,
    std_reward=1.5,
    penalty_fn=linear_penalty
)

generators = [
    TaskGenerator(quick_tasks, generator_seed=10, probability=0.3),
    TaskGenerator(big_tasks, generator_seed=11, probability=0.1)
]

env = Environment(generators=generators, timesteps=1000)
print(env.all_tasks)
for agent in [RandomAgent(seed=42, horizon=10), GreedyAgent(horizon=10)]:
    sim = Simulator(env=env, agent=agent, horizon=10)
    results = sim.run()
    summarise(agent.name, results, env)