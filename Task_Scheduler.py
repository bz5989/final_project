class TaskScheduler:

    def __init__(self, H):
        self.H = H
        self.schedule_window = [0] * H
        self.allocation = {}

    def reset(self):
        self.schedule_window = [0] * self.H
        self.allocation = {}
        return self.schedule_window

    def get_schedule_window(self):
        return self.schedule_window

    def shift(self):
        job = self.schedule_window[0]
        self.schedule_window = self.schedule_window[1:] + [0]
        return job
    
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
        
        self.allocation[job_id] = self.allocation.get(job_id, 0) + pred_length
    
    def get_allocation(self, job_id):
        return self.allocation.get(job_id, 0)

    def copy(self):
        s = TaskScheduler(self.H)
        s.schedule_window = self.schedule_window.copy()
        s.allocation = self.allocation.copy()
        return s