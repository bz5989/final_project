import random

class Environment:
    def __init__(self, H, alpha, beta):
        self.H = H
        self.alpha = alpha
        self.beta = beta
        self.jobs = {}
        self.t = 0

    def sample_job(self):
        if random.random() < 0.5:
            return None

        return {
            "id": random.randint(1, 99999),
            "type": random.randint(1, 3),
            "length": random.randint(1, 2),
            "deadline": random.randint(2, self.H - 1),
            "progress": 0,
            "complete": False
        }

    def add_job(self, job):
        self.jobs[job["id"]] = job

    def disruption(self, old, new):
        return sum(int(a != b) for a, b in zip(old, new))

    def lateness_reward(self):
        reward = 0
        for job in self.jobs.values():
            if not job["complete"] and self.t > job["deadline"]:
                reward -= self.beta
                job["complete"] = True
        return reward

    def step_time(self):
        self.t += 1
