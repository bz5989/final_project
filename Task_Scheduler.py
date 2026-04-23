class TaskScheduler:

    def __init__(self, H):
        self.H = H
        self.schedule_window = [0] * H

    def reset(self):
        self.schedule_window = [0] * self.H
        return self.schedule_window

    def get_schedule_window(self):
        return self.schedule_window

    def shift(self):
        self.schedule_window = self.schedule_window[1:] + [0]

    def can_place(self, i, length):
        if i + length > self.H:
            return False
        return all(self.schedule_window[i + k] == 0 for k in range(length))

    def place(self, job, i):
        jid = job["id"]
        original = []
        for k in range(job["length"]):
            original.append(self.schedule_window[i + k])
            self.schedule_window[i + k] = jid
        return original

    def copy(self):
        s = TaskScheduler(self.H)
        s.schedule_window = self.schedule_window.copy()
        return s