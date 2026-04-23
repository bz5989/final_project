import torch
import torch.nn as nn
import torch.nn.functional as F
from Task_Scheduler import TaskScheduler
from pseudo_environment import Environment

class PPO_Pointer_Network(nn.Module):
    def __init__(self, H, emb=64, hid=128):
        super().__init__()
        self.H = H
        self.embedding = nn.Embedding(100, emb)
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
            pred_duration = torch.ones(B)
        else:
            job_vec = self.job_encoder(job)
            pred_duration = self.duration_head(job_vec).squeeze(-1)

        job_query = self.W_query(job_vec).unsqueeze(1)
        schedule_meaning = self.W_ref(enc)

        u = torch.tanh(schedule_meaning + job_query)
        logits = torch.matmul(u, self.v)

        if mask is not None:
            logits = logits + (mask - 1) * 1e9

        value = self.value_head(enc.mean(dim=1)).squeeze(-1)
        return logits, value, pred_duration

    @torch.no_grad()
    def get_action(self, schedule, job, mask=None):
        logits, value, pred_duration = self.forward(schedule, job, mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value, pred_duration

    def get_log_prob(self, schedule, job, action, mask=None):
        logits, value, pred_duration = self.forward(schedule, job, mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action), value, pred_duration

def collect_batch(env, scheduler, model, num_steps=100, gamma=0.99, lam=0.95):
    schedules, actions, log_probs = [], [], []
    rewards, values, jobs, masks = [], [], [], []
    pred_durations, true_lengths = [], []
    
    allocation = {}

    schedule = scheduler.reset()
    env.t = 0

    for _ in range(num_steps):
        job = env.sample_job()
        mask = scheduler.valid_mask(job)
        job_tensor = torch.tensor([job["type"], job["deadline"], job["reward"]]) if job else torch.zeros(3)
        action, log_prob, value, pred_dur = model.get_action(torch.tensor(schedule),
                                                            job_tensor, 
                                                            torch.tensor(mask))
        
        pred_length = int(torch.clamp(torch.round(pred_dur), 1, scheduler.H).item())
        
        reward = 0

        if job:
            jid = job["id"]
            if scheduler.can_place(action, pred_length):
                scheduler.place({"id": jid, "length": pred_length}, action)
                env.add_job(job)

                allocation[jid] = allocation.get(jid, 0) + pred_length

                pred_durations.append(pred_dur)
                true_lengths.append(torch.tensor(job["length"], dtype=torch.float32))
            else:
                reward -= 1

        finished = scheduler.shift()
        env.step_time()

        if finished != 0:
            jid = finished
            if jid in env.jobs:
                true_length = env.jobs[jid]["length"]
                alloc = allocation.get(jid, 0)
                if alloc < true_length:
                    reward -= (true_length - alloc)

        reward += env.lateness_reward()
        schedule = scheduler.get_schedule_window()

        schedules.append(torch.tensor(schedule))
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        jobs.append(job_tensor)
        masks.append(torch.tensor(mask))
    
    job = env.sample_job()
    mask = scheduler.valid_mask(job)
    _, _, value = model.get_action(torch.tensor(schedule), job, mask)
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
            new_log_probs, values, pred_duration = model.get_log_prob(schedules[batch_indices], jobs[batch_indices], 
                                           actions[batch_indices], masks[batch_indices])
            
            ratio = torch.exp(new_log_probs - old_logp[batch_indices])
            s1 = ratio * advantages[batch_indices]
            s2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages[batch_indices]

            policy_loss = -torch.mean(torch.min(s1, s2))
            value_loss = nn.mse_loss(values, returns[batch_indices])
            dur_loss = F.mse_loss(pred_durations, true_lengths)

            loss = policy_loss + value_coef * value_loss + dur_coef * dur_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_ppo():
    H = 8
    model = PPO_Pointer_Network(H)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = TaskScheduler(H)
    env = Environment(H, 1, 2)

    for ep in range(200):
        batch = collect_batch(env, scheduler, model)
        ppo_update(model, opt, batch)

        if ep % 20 == 0:
            print("Episode", ep)


if __name__ == "__main__":
    train_ppo()