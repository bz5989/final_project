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
                self.insert_task(job, timestep)
        # self.shift()
        schedule = self.get_schedule_window()
        schedule.shift()
        return schedule

    def reset(self):
        self.schedule_window = [0] * self.H
        self.job_ids = {}

    def get_schedule_window(self):
        return self.schedule_window

    def insert_task(self, job, current_time):
        # change to encode based on types
        schedule_tensor = torch.tensor(self.schedule_window)
        job_tensor = torch.tensor(self.encode_job(job, current_time))
        pred_length = self.model.get_pred_length(job_tensor)
        mask_tensor = torch.tensor(self.valid_mask(pred_length))

        action, _, _ = self.model.get_action(schedule_tensor, job_tensor, mask_tensor)

        action_item = action.item()

        self.place(job.instance_id, pred_length, action_item)

        self.job_ids[job.instance_id] = job

    def encode_job(self, job, current_time):
        return [job.task_type if hasattr(job, "task_type") else 1,
            max(0, float(job.deadline - current_time)) if hasattr(job, "deadline") else float(self.H),
            job.reward if hasattr(job, "reward") else 0]

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