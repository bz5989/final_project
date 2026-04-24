import torch
import torch.nn as nn
import torch.nn.functional as F
from PPOSchedulerAgent import PPOSchedulerAgent
from Environment import Environment
from Core import TaskCategory, TaskGenerator, linear_penalty

class PPO_Pointer_Network(nn.Module):
    def __init__(self, H, emb=3, hid=128):
        super().__init__()
        self.H = H
        
        self.embedding = nn.Embedding(H, emb)
        self.encoder = nn.LSTM(emb, hid, batch_first=True)

        self.job_encoder = nn.Sequential(
            nn.Linear(3, hid), 
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
        return self.encoder(self.embedding(s.unsqueeze(0)))[0].squeeze(0)

    def forward(self, schedule, job, mask=None):
        enc = self.encode(schedule)

        if enc.dim() == 2:
            enc = enc.unsqueeze(0)

        B = enc.size(0)

        if job is None:
            job_vec = torch.zeros(B, enc.size(-1))
        else:
            job_vec = self.job_encoder(job)

        job_query = self.W_query(job_vec).unsqueeze(1)
        schedule_meaning = self.W_ref(enc)

        u = torch.tanh(schedule_meaning + job_query)
        logits = torch.matmul(u, self.v)

        if mask is not None:
            logits = logits + (mask - 1) * 1e9

        value = self.value_head(enc.mean(dim=1)).squeeze(-1)
        return logits, value
    
    def get_pred_duration(self, job):
        if job is None:
            return 0
        job_vec = self.job_encoder(job)
        pred_duration = self.duration_head(job_vec).squeeze(-1)
        return pred_duration
    
    def get_pred_length(self, job):
        pred_dur = self.get_pred_duration(job)
        pred_length = int(torch.ceil(pred_dur).item())
        return pred_length

    @torch.no_grad()
    def get_action(self, schedule, job, mask=None):
        logits, value = self.forward(schedule, job, mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def get_log_prob(self, schedule, job, action, mask=None):
        logits, value = self.forward(schedule, job, mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), value

def collect_batch(env, scheduler, model, num_steps=100, gamma=0.99, lam=0.95):
    schedules, actions, log_probs = [], [], []
    rewards, values, jobs, masks = [], [], [], []
    pred_durations, true_lengths = [], []
    
    allocation = {}

    schedule = scheduler.reset()
    env.t = 0

    for _ in range(num_steps):
        schedule_tensor = torch.tensor(scheduler.embed_schedule())
        job = env.sample_job()
        job_tensor = torch.tensor([job.task_category, job.deadline_time, job.base_reward]) if job else torch.zeros(3)
        pred_dur = model.get_pred_duration(job_tensor)
        pred_length = int(torch.ceil(pred_dur).item())
        mask_tensor = torch.tensor(scheduler.valid_mask(pred_length))
        action, log_prob, value = model.get_action(schedule_tensor, job_tensor, mask_tensor)
        # action is equivalent to location where placed
        reward = 0

        if job:
            jid = job.instance_id
            if scheduler.can_place(action, pred_length):
                scheduler.place(jid, pred_length, action)

                allocation[jid] = allocation.get(jid, 0) + pred_length

                pred_durations.append(pred_dur)
                true_lengths.append(torch.tensor(job.duration, dtype=torch.float32))
            else:
                reward -= job.base_reward

        finished = scheduler.shift()
        env.step_time()

        # penalize underestimates of jobs
        if finished != 0:
            jid = finished
            if jid in env.jobs:
                true_length = env.jobs[jid]["length"]
                alloc = allocation.get(jid, 0)
                if alloc < true_length:
                    reward -= (true_length - alloc)

        reward += env.lateness_reward()
        schedule = scheduler.get_schedule_window()

        schedules.append(schedule_tensor)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        jobs.append(job_tensor)
        masks.append(mask_tensor)
    
    job = env.sample_job()
    mask_tensor = torch.tensor(scheduler.valid_mask(job))
    _, _, value = model.get_action(torch.tensor(schedule), job, mask_tensor)
    values.append(value)

    advantages = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    returns = [a+v for a,v in zip(advantages, values[:-1])]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch = {
        'schedules': schedules,
        'actions': actions,
        'log_probs': log_probs,
        'jobs': jobs,
        'masks': masks,
        'returns': returns,
        'advantages': advantages,
        'pred_durations': pred_durations,
        'true_lengths': true_lengths
    }

    return batch

def ppo_update(model, optimizer, batch, clip_epsilon=0.2, epochs=4, batch_size=32, value_coef=0.5, dur_coef=0.1):
    schedules = torch.stack(batch["schedules"])
    actions = torch.tensor(batch["actions"])
    old_logp = torch.tensor(batch["log_probs"])
    jobs = torch.stack(batch["jobs"])
    masks = torch.tensor(batch["masks"])
    returns = torch.tensor(batch["returns"])
    advantages = torch.tensor(batch["advantages"])

    pred_durations = torch.stack(batch["pred_durations"])
    true_lengths = torch.stack(batch["true_lengths"])

    idx = torch.randperm(schedules)

    for _ in range(epochs):
        for i in range(0, len(schedules), batch_size):
            batch_indices = idx[i:i + batch_size]
            new_log_probs, values = model.get_log_prob(schedules[batch_indices], jobs[batch_indices], 
                                           actions[batch_indices], masks[batch_indices])
            
            ratio = torch.exp(new_log_probs - old_logp[batch_indices])
            s1 = ratio * advantages[batch_indices]
            s2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages[batch_indices]

            policy_loss = -torch.mean(torch.min(s1, s2))
            value_loss = nn.mse_loss(values, returns[batch_indices])
            dur_loss = nn.mse_loss(pred_durations, true_lengths)

            loss = policy_loss + value_coef * value_loss + dur_coef * dur_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_ppo(model, optimizer, scheduler, env):
    for ep in range(200):
        batch = collect_batch(env, scheduler, model)
        ppo_update(model, optimizer, batch)

        if ep % 20 == 0:
            print("Episode", ep)


if __name__ == "__main__":
    H = 8
    model = PPO_Pointer_Network(H)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = PPOSchedulerAgent(model, H)


    # environment taken from runner.py
    quick_tasks = TaskCategory(
        name="Quick",
        category_seed=1,
        mean_time=1,
        std_time=0.3,
        mean_reward=3.0,
        std_reward=0.5,
        penalty_fn=linear_penalty
    )

    big_tasks = TaskCategory(
        name="Big",
        category_seed=2,
        mean_time=5,
        std_time=1.0,
        mean_reward=10.0,
        std_reward=1.5,
        penalty_fn=linear_penalty
    )

    generators = [
        TaskGenerator(quick_tasks, generator_seed=10, probability=0.3),
        TaskGenerator(big_tasks, generator_seed=11, probability=0.1)
    ]
    env = Environment(generators=generators, timesteps=1000)
    train_ppo(model, optimizer, scheduler, env)