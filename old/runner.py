from classes import SchedulerAgent, AssignmentGenerator, Assignment, Type
from torch.distributions import Normal

if __name__ == "__main__":
    types = {
        'pset_short': {
            'label': 'pset_short',
            'true_progress_distribution': Normal(5.0, 1.0),
            'flow_state_distribution': Normal(1.0, 0.3),
            'drop_off_rate': 0.05,
            'rate': 1.0 / 24.0,
        },
        'pset_long': {
            'label': 'pset_long',
            'true_progress_distribution': Normal(15.0, 3.0),
            'flow_state_distribution': Normal(1.0, 0.1),
            'drop_off_rate': 0.05,
            'rate': 1.0 / 24.0 / 3.0,
        },
        'project': {
            'label': 'project',
            'true_progress_distribution': Normal(100.0, 5.0),
            'flow_state_distribution': Normal(1.0, 0.1),
            'drop_off_rate': 0.05,
            'rate': 1.0 / 24.0 / 21.0,
        },
    }
    generator = AssignmentGenerator(types)
    schedule = SchedulerAgent('my_schedule', types)