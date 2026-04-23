import torch

class PPOSchedulerAgent:
    name="Scheduler"
    def __init__(self, model, horizon):
        self.H = horizon
        self.model = model
        self.schedule_window = [None] * self.H
        self.task_map = {}

    def plan(self, available_tasks, timestep):
        scheduled_ids = {t.instance_id for t in self.task_map.values()}

        for task in available_tasks:
            if task.instance_id not in scheduled_ids:
                self.insert_task(task, timestep)

        focus = self.schedule_window[0]
        self.shift()

        return [focus] if focus else [None]

    def reset(self):
        self.schedule_window = [None] * self.H
        self.task_map = {}

    def insert_task(self, task, current_time):
        job_vec = self.encode_job(task, current_time)
        schedule_tensor = self.encode_schedule()
        mask = self.valid_mask(task.duration)

        action, _, _, _ = self.model.get_action(
            schedule_tensor, job_vec, mask
        )

        action_item = action.item()

        # place task
        for k in range(task.duration):
            self.schedule_window[action_item + k] = task

        self.task_map[task.instance_id] = task

    def encode_job(self, task, current_time):
        return torch.tensor([
            task.task_type if hasattr(task, "task_type") else 1,
            float(task.duration),
            max(0, float(task.deadline - current_time)) if hasattr(task, "deadline") else float(self.H)
        ], dtype=torch.float32)

    def encode_schedule(self):
        ids = [
            0 if x is None else (x.instance_id % 100)
            for x in self.schedule_window
        ]
        return torch.tensor(ids, dtype=torch.long)

    def valid_mask(self, length):
        mask = []
        for i in range(self.H):
            if i + length > self.H:
                mask.append(0)
            else:
                ok = all(self.schedule_window[i + k] is None for k in range(length))
                mask.append(1 if ok else 0)
        return torch.tensor(mask, dtype=torch.float32)

    def shift(self):
        self.schedule_window = self.schedule_window[1:] + [None]