import torch
import torch.nn as nn
import torch.nn.functional as F
from PPOSchedulerAgent import PPOSchedulerAgent
from Environment import Environment
from utils import plot_diagnostics
from Core import TaskCategory, TaskGenerator, linear_penalty

class PPO_Pointer_Network(nn.Module):
    def __init__(self, H, emb=3, hid=128):
        super().__init__()
        self.H = H
        
        self.encoder = nn.LSTM(input_size=2, hidden_size=hid, batch_first=True)

        self.job_encoder = nn.Sequential(
            nn.Linear(2, hid), 
            nn.ReLU(), 
            nn.Linear(hid, hid)
        )

        self.W_query = nn.Linear(hid, hid)
        self.W_ref = nn.Linear(hid, hid)
        self.v = nn.Parameter(torch.randn(hid))

        self.value_head = nn.Linear(hid, 1)

        self.duration_head = nn.Sequential(
            nn.Linear(hid, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
            nn.Softplus()  # ensures positive
        )

    def encode(self, s):
        if s.dim() == 1:
            s = s.unsqueeze(0)
        
        enc, _ = self.encoder(s)
        return enc

    def forward(self, schedule, job, mask=None):
        enc = self.encode(schedule)
        job_vec = self.job_encoder(job)

        schedule_meaning = self.W_ref(enc)
        job_query = self.W_query(job_vec).unsqueeze(-2)
        # print(enc.shape, job_vec.shape)
        # print(schedule_meaning.shape, job_query.shape)

        u = torch.tanh(schedule_meaning + job_query)
        logits = torch.matmul(u, self.v)

        if mask is not None:
            mask = mask.float()
            logits = logits + (mask - 1) * 1e9

        return logits
    
    def get_value(self, schedule):
        enc = self.encode(schedule)
        pool = enc.mean(dim=1)
        value = self.value_head(pool).squeeze(-1)
        return value

    def get_pred_duration(self, job):
        if job.dim() == 1:
            job = job.unsqueeze(0)

        job_vec = self.job_encoder(job)
        pred_duration = self.duration_head(job_vec).squeeze(-1)
        return pred_duration
    
    def get_pred_length(self, job):
        pred_dur = self.get_pred_duration(job)
        pred_length = int(torch.ceil(pred_dur).item())
        return pred_length

    @torch.no_grad()
    def get_action(self, schedule, job, mask=None):
        logits = self.forward(schedule, job, mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def get_log_prob(self, schedule, job, action, mask=None):
        logits = self.forward(schedule, job, mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action)

def collect_batch(env, scheduler, model, num_steps=100, gamma=0.99, lam=0.95):
    schedules, actions, log_probs = [], [], []
    rewards, values, jobs, masks = [], [], [], []
    pred_durations, true_lengths = [], []
    valid = []

    scheduler.reset()

    for _ in range(num_steps):
        schedule_tensor = torch.tensor(scheduler.embed_schedule(), dtype=torch.float32)
        id, job = env.sample_job()

        reward = 0
        value = model.get_value(schedule_tensor.unsqueeze(0)).squeeze(0).detach()

        job_tensor = torch.tensor([-1, 0])
        action = torch.tensor(0)
        log_prob = torch.tensor(0.0)
        mask_tensor = torch.zeros_like(torch.tensor(scheduler.valid_mask(1)))

        if job:
            # print(job.task_id)
            job_tensor = torch.tensor([job.task_id, job.deadline_time - scheduler.t])

            pred_dur = model.get_pred_duration(job_tensor)
            pred_length = int(torch.ceil(pred_dur).item())

            mask_tensor = torch.tensor(scheduler.valid_mask(pred_length))
            action, log_prob = model.get_action(schedule_tensor, job_tensor, mask_tensor)

            # action is equivalent to location where placed
            if scheduler.can_place(action, pred_length):
                scheduler.place(job, pred_length, action)
            else:
                reward -= job.base_reward
            
            pred_durations.append(pred_dur.detach())
            true_lengths.append(torch.tensor(job.duration_time, dtype=torch.float32))
            valid.append(1)
        else:
            pred_durations.append(torch.tensor([0.0]))
            true_lengths.append(torch.tensor(0.0))
            valid.append(0)
        
        worked_job = scheduler.shift()
        env.step_time()

        # penalize underestimates of jobs
        if worked_job != -1:
            reward += scheduler.check_reward(worked_job)

        # reward += env.lateness_reward()

        schedules.append(schedule_tensor)
        actions.append(action.squeeze())
        log_probs.append(log_prob.squeeze())
        rewards.append(reward)
        values.append(value)
        jobs.append(job_tensor)
        masks.append(mask_tensor)
    
    schedule_tensor = torch.tensor(scheduler.embed_schedule(), dtype=torch.float32)
    value = model.get_value(schedule_tensor.unsqueeze(0)).squeeze(0).detach()
    values.append(value)

    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    returns = [a+v for a,v in zip(advantages, values[:-1])]

    advantages = torch.tensor(advantages)
    returns = torch.tensor(returns)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch = {
        'schedules': torch.stack(schedules),
        'actions': torch.stack(actions),
        'log_probs': torch.stack(log_probs),
        'jobs': torch.stack(jobs),
        'masks': torch.stack(masks),
        'returns': returns,
        'advantages': advantages,
        'pred_durations': torch.stack(pred_durations),
        'true_lengths': torch.stack(true_lengths),
        'valid': torch.tensor(valid)
    }

    return batch

def ppo_update(model, optimizer, batch, clip_epsilon=0.2, epochs=4, batch_size=32, value_coef=0.5, dur_coef=0.1):
    schedules = batch["schedules"]
    actions = batch["actions"]
    old_logp = batch["log_probs"]
    jobs = batch["jobs"]
    masks = batch["masks"]
    returns = batch["returns"]
    advantages = batch["advantages"]

    pred_durations = batch["pred_durations"]
    true_lengths = batch["true_lengths"]

    valid = batch["valid"]

    policy_losses = 0.0
    value_losses = 0.0
    dur_losses = 0.0

    num_batches = len(schedules) / batch_size
    for _ in range(epochs):
        idx = torch.randperm(len(schedules))
        for i in range(0, len(schedules), batch_size):
            batch_indices = idx[i:i + batch_size]
            new_log_probs = model.get_log_prob(schedules[batch_indices], jobs[batch_indices], 
                                           actions[batch_indices], masks[batch_indices])
            
            new_durations = model.get_pred_duration(jobs[batch_indices])
            values = model.get_value(schedules[batch_indices])

            ratio = torch.exp(new_log_probs - old_logp[batch_indices])
            s1 = ratio * advantages[batch_indices]
            s2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages[batch_indices]

            policy_loss = -torch.mean(torch.min(s1, s2))
            value_loss = F.mse_loss(values, returns[batch_indices])
            dur_loss = F.mse_loss(new_durations, true_lengths[batch_indices])

            loss = policy_loss + value_coef * value_loss + dur_coef * dur_loss

            policy_losses += policy_loss.item()
            value_losses += value_loss.item()
            dur_losses += dur_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return {'policy_loss': policy_losses / epochs / num_batches,
            'value_loss': value_losses / epochs / num_batches,
            'dur_loss': dur_losses / epochs / num_batches}

def train_ppo(model, optimizer, scheduler, env):
    # training_rewards = []
    metrics = {'policy_loss': [], 'value_loss': [], 'dur_loss': []}
    for ep in range(200):
        batch = collect_batch(env, scheduler, model, num_steps=200)
        dict = ppo_update(model, optimizer, batch)
        
        for key in metrics:
            metrics[key].append(dict[key])

        if ep % 20 == 0:
            print("Episode", ep)

    return metrics

import random
class MicroEnv:
    def __init__(self, env):
        self.time = 0
        self.jobs = 0
        self.env = env
    
    def sample_job(self):
        items = self.env.get_timestep(self.time).task_instances
        if len(items) == 0:
            return -1, None
        elif len(items) == 1:
            self.jobs += 1
            return self.jobs - 1, items[0]
        rand = random.randint(0, len(items) - 1)
        self.jobs += 1
        return self.jobs - 1, items[rand]
    
    def step_time(self):
        self.time += 1

if __name__ == "__main__":
    H = 8
    model = PPO_Pointer_Network(H)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = PPOSchedulerAgent(model, H)


    # environment taken from runner.py
    quick_tasks = TaskCategory(
        name="Quick",
        task_id=1,
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
        task_id=2,
        category_seed=2,
        mean_time=5,
        std_time=1.0,
        mean_buffer_time=1.0,
        std_buffer_time=0.4,
        mean_reward=10.0,
        std_reward=1.5,
        penalty_fn=linear_penalty
    )

    generators = [
        TaskGenerator(quick_tasks, generator_seed=10, probability=0.3),
        TaskGenerator(big_tasks, generator_seed=11, probability=0.1)
    ]
    env = Environment(generators=generators, timesteps=40000)
    env_wrapper = MicroEnv(env)
    metrics = train_ppo(model, optimizer, scheduler, env_wrapper)

    plot_diagnostics(metrics, "Test", "diagnostics.png")