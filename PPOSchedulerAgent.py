import torch

class PPOSchedulerAgent:
    name="Scheduler"
    def __init__(self, model, horizon):
        self.H = horizon
        self.model = model
        self.schedule_window = [0] * self.H
        self.task_ids = {}

    def plan(self, available_tasks, timestep):

        for task in available_tasks:
            if task.instance_id not in self.tasks.keys():
                self.insert_task(task, timestep)

        self.shift()
        return self.get_schedule_window()

    def reset(self):
        self.schedule_window = [0] * self.H
        self.task_ids = {}

    def get_schedule_window(self):
        return self.schedule_window

    def insert_task(self, task, current_time):
        schedule_tensor = self.encode_schedule()
        job_tensor = torch.tensor(self.encode_job(task, current_time))
        mask_tensor = torch.tensor(self.valid_mask(task.duration))

        pred_length = self.model.get_pred_length(job_tensor)
        action, _, _ = self.model.get_action(schedule_tensor, job_tensor, mask_tensor)

        action_item = action.item()

        self.place(task.instance_id, pred_length, action_item)

        self.task_ids[task.instance_id] = task

    def encode_job(self, task, current_time):
        return [task.task_type if hasattr(task, "task_type") else 1,
            float(task.duration),
            max(0, float(task.deadline - current_time)) if hasattr(task, "deadline") else float(self.H)]

    def encode_schedule(self):
        ids = [id % 100 for id in self.schedule_window]
        return torch.tensor(ids, dtype=torch.long)

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
        s.task_ids = self.task_ids.copy()
        return s