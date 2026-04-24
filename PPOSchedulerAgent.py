import torch

class PPOSchedulerAgent:
    name="Scheduler"
    def __init__(self, model, horizon):
        self.H = horizon
        self.model = model
        self.schedule_window = [0] * self.H
        self.job_ids = {}

    def plan(self, available_jobs, timestep):

        for job in available_jobs:
            if job.instance_id not in self.job_ids.keys():
                self.insert_task(job)
        # self.shift()
        schedule = self.get_schedule_window()
        schedule.shift()
        return schedule

    def reset(self):
        self.schedule_window = [0] * self.H
        self.job_ids = {}

    def get_schedule_window(self):
        return self.schedule_window

    def embed_schedule(self):
        schedule_embedding = [[0, 0, 0] if x == 0 else [self.job_ids[x][0].task_category, 
                                                        self.job_ids[x][0].deadline_time, 
                                                        self.job_ids[x][0].base_reward] for x in self.schedule_window]
        return schedule_embedding
    
    def insert_task(self, job):
        # change to encode based on types
        schedule_tensor = torch.tensor(self.embed_schedule())
        job_tensor = torch.tensor(self.encode_job(job))
        pred_length = self.model.get_pred_length(job_tensor)
        mask_tensor = torch.tensor(self.valid_mask(pred_length))

        action, _, _ = self.model.get_action(schedule_tensor, job_tensor, mask_tensor)

        action_item = action.item()

        self.place(job.instance_id, pred_length, action_item)

        self.job_ids[job.instance_id] = [job, pred_length]

    def encode_job(self, job):
        job_vals = [job.task_category, job.deadline_time, job.base_reward] if job else [0, 0, 0]
        return job_vals

    def valid_mask(self, pred_length):
        mask = []
        for i in range(self.H):
            if i + pred_length > self.H:
                mask.append(0)
            else:
                ok = all(self.schedule_window[i + k] == 0 for k in range(pred_length))
                mask.append(1 if ok else 0)
        return mask

    def can_place(self, i, pred_length):
        if i + pred_length > self.H:
            return False
        return all(self.schedule_window[i + k] == 0 for k in range(pred_length))

    def place(self, job_id, pred_length, i):
        for k in range(pred_length):
            self.schedule_window[i + k] = job_id

    def shift(self):
        job = self.schedule_window[0]
        self.schedule_window = self.schedule_window[1:] + [0]
        return job
    
    def copy(self):
        s = PPOSchedulerAgent(self.model, self.H)
        s.schedule_window = self.schedule_window.copy()
        s.job_ids = self.job_ids.copy()
        return s