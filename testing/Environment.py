"""
scheduler_env.py
~~~~~~~~~~~~~~~~
A task-scheduling Gymnasium environment with a configurable number of task
categories and a rolling planning horizon.

Horizon mechanics
-----------------
self._horizon always holds the current committed plan (length = horizon_len).

Each call to step(action):
  1. Penalty  = penalty_coeff * |{i : action[i] != self._horizon[i]}|
                (cost of deviating from the previously committed plan)
  2. Commit   : self._horizon <- action
  3. Execute  : work on self._horizon[0]; collect reward if task completes
  4. Shift    : self._horizon[:-1] = self._horizon[1:], self._horizon[-1] = 0

Task reward
-----------
Base reward is sampled at arrival.  On completion the reward has decayed by
  decay_rate * (completion_time - start_time)
clamped to zero.  decay_rate is sampled per task from the category's
decay_range.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import numpy as np
import gymnasium as gym


# ── Task category definition ───────────────────────────────────────────────────

@dataclass
class TaskCategory:
    """
    Parameters that fully define one category of task.

    Attributes
    ----------
    arrival_prob   : probability [0, 1] a task arrives each timestep.
    duration_range : (lo, hi) inclusive — units of work needed.
    buffer_range   : (lo, hi) inclusive — slack added on top of duration
                     to compute the deadline.
    reward_range   : (lo, hi) — base reward sampled uniformly.
    decay_range    : (lo, hi) — per-step reward decay sampled uniformly.
                     Reward at completion = max(0, base - decay*(t - t_start)).
    """
    arrival_prob:   float
    duration_range: tuple[int,   int]
    buffer_range:   tuple[int,   int]
    reward_range:   tuple[float, float]
    decay_range:    tuple[float, float] = field(default=(0.0, 0.0))

    def __post_init__(self) -> None:
        assert 0.0 <= self.arrival_prob <= 1.0
        assert self.duration_range[0] <= self.duration_range[1]
        assert self.buffer_range[0]   <= self.buffer_range[1]
        assert self.reward_range[0]   <= self.reward_range[1]
        assert self.decay_range[0]    <= self.decay_range[1]


# Default two-category setup from the original spec
DEFAULT_CATEGORIES: list[TaskCategory] = [
    TaskCategory(
        arrival_prob=0.10,
        duration_range=(5, 12), buffer_range=(3, 6),
        reward_range=(10.0, 20.0), decay_range=(0.1, 0.3),
    ),
    TaskCategory(
        arrival_prob=0.05,
        duration_range=(1,  6),  buffer_range=(1, 2),
        reward_range=( 5.0, 10.0), decay_range=(0.05, 0.15),
    ),
]


# ── Environment ────────────────────────────────────────────────────────────────

class SchedulerEnv(gym.Env):
    """
    Task scheduling environment with configurable categories and a rolling
    planning horizon.

    Parameters
    ----------
    categories      : one TaskCategory per task type.
    max_timesteps   : episode length; also caps tasks-per-category.
    horizon_len     : number of future steps planned at once.
    max_reward      : upper bound on any single task reward (for spaces).
    penalty_coeff   : per-slot penalty for deviating from the previous plan.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        categories:    list[TaskCategory] = DEFAULT_CATEGORIES,
        max_timesteps: int   = 1_000,
        horizon_len:   int   = 50,
        max_reward:    float = 1_000.0,
        penalty_coeff: float = 0.5,
    ) -> None:
        super().__init__()

        self.categories     = categories
        self.max_timesteps  = max_timesteps
        self.horizon_len    = horizon_len
        self.max_reward     = max_reward
        self.penalty_coeff  = penalty_coeff
        self.num_categories = len(categories)

        # ID space: block of max_timesteps IDs per category; 0 = idle
        self._max_id: int = self.num_categories * max_timesteps

        # ── Spaces ────────────────────────────────────────────────────────
        _task_space = gym.spaces.Dict({
            "id":         gym.spaces.Discrete(self._max_id + 1),
            "category":   gym.spaces.Discrete(self.num_categories),
            "start_time": gym.spaces.Discrete(max_timesteps),
            "end_time":   gym.spaces.Discrete(max_timesteps),
            "duration":   gym.spaces.Discrete(max_timesteps),
            "reward":     gym.spaces.Box(0.0, max_reward, shape=(1,), dtype=np.float32),
            "decay_rate": gym.spaces.Box(0.0, max_reward, shape=(1,), dtype=np.float32),
        })

        self.observation_space = gym.spaces.Dict({
            "new_tasks":       gym.spaces.Sequence(_task_space),
            "current_horizon": gym.spaces.MultiDiscrete(
                                   np.full(horizon_len, self._max_id + 1, dtype=np.int64)
                               ),
            "previous_task":   gym.spaces.Discrete(self._max_id + 1),
        })

        # Action = proposed new horizon (replaces self._horizon before shift)
        self.action_space = gym.spaces.MultiDiscrete(
            np.full(horizon_len, self._max_id + 1, dtype=np.int64)
        )

        # Runtime state — reinitialised in reset()
        self.t:              int          = 0
        self._counters:      list[int]    = [0] * self.num_categories
        self._task_pool:     dict[int, dict] = {}
        self._horizon:       np.ndarray   = np.zeros(horizon_len,        dtype=np.int32)
        self._agent_history: np.ndarray   = np.zeros(max_timesteps,      dtype=np.int32)
        self._task_history:  np.ndarray   = np.zeros((max_timesteps, 2), dtype=np.int32)

    # ── Task ID helpers ────────────────────────────────────────────────────

    def _next_id(self, cat_index: int) -> int:
        self._counters[cat_index] += 1
        if self._counters[cat_index] > self.max_timesteps:
            raise RuntimeError(f"ID pool exhausted for category {cat_index}.")
        return cat_index * self.max_timesteps + self._counters[cat_index]

    # ── Task generation ────────────────────────────────────────────────────

    def _gen_task(self, cat_index: int) -> dict:
        cat      = self.categories[cat_index]
        duration = int(self.np_random.integers(cat.duration_range[0],
                                               cat.duration_range[1] + 1))
        buffer   = int(self.np_random.integers(cat.buffer_range[0],
                                               cat.buffer_range[1] + 1))
        reward     = float(self.np_random.uniform(*cat.reward_range))
        decay_rate = float(self.np_random.uniform(*cat.decay_range))
        return {
            "id":         self._next_id(cat_index),
            "category":   cat_index,
            "start_time": self.t,
            "end_time":   min(self.max_timesteps - 1, self.t + duration + buffer),
            "duration":   duration,
            "reward":     np.array([reward],     dtype=np.float32),
            "decay_rate": np.array([decay_rate], dtype=np.float32),
        }

    def _gen_tasks(self) -> list[dict]:
        arriving = []
        for i, cat in enumerate(self.categories):
            if self.np_random.uniform() < cat.arrival_prob:
                arriving.append(self._gen_task(i))
        return arriving

    # ── Pool maintenance ───────────────────────────────────────────────────

    def _register_tasks(self, tasks: list[dict]) -> None:
        for task in tasks:
            self._task_pool[task["id"]] = task

    def _prune_overdue(self) -> None:
        expired = [tid for tid, t in self._task_pool.items() if t["end_time"] < self.t]
        for tid in expired:
            del self._task_pool[tid]

    # ── Observation / info ─────────────────────────────────────────────────

    def _get_obs(self, new_tasks: list[dict]) -> dict:
        previous = int(self._agent_history[self.t - 1]) if self.t > 0 else 0
        return {
            "new_tasks":       tuple(new_tasks),
            "current_horizon": self._horizon.copy(),
            "previous_task":   previous,
        }

    def _get_info(self, completed: Optional[dict] = None, penalty: float = 0.0) -> dict:
        return {
            "t":              self.t,
            "pool_size":      len(self._task_pool),
            "pool_ids":       list(self._task_pool.keys()),
            "completed_task": completed,
            "penalty":        penalty,
        }

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[dict] = None,
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)

        self.t              = 0
        self._counters      = [0] * self.num_categories
        self._task_pool     = {}
        self._horizon       = np.zeros(self.horizon_len,        dtype=np.int32)
        self._agent_history = np.zeros(self.max_timesteps,      dtype=np.int32)
        self._task_history  = np.zeros((self.max_timesteps, 2), dtype=np.int32)

        new_tasks = self._gen_tasks()
        self._register_tasks(new_tasks)
        return self._get_obs(new_tasks), self._get_info()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """
        Parameters
        ----------
        action : int array of shape (horizon_len,)
            Proposed new horizon.  action[0] will be executed this step.

        Horizon update sequence
        -----------------------
        1. penalty  <- penalty_coeff * number of slots that differ from self._horizon
        2. self._horizon <- action
        3. Execute self._horizon[0]
        4. Shift left: self._horizon[:-1] = self._horizon[1:], self._horizon[-1] = 0
        """
        action = np.asarray(action, dtype=np.int32)
        assert action.shape == (self.horizon_len,), (
            f"Expected action shape ({self.horizon_len},), got {action.shape}"
        )

        # ── 1. Deviation penalty ───────────────────────────────────────────
        deviations = int(np.sum(action != self._horizon))
        penalty    = self.penalty_coeff * deviations

        # ── 2. Commit new horizon ──────────────────────────────────────────
        self._horizon = action.copy()

        # ── 3. Execute horizon[0] ──────────────────────────────────────────
        task_id   = int(self._horizon[0])
        reward    = 0.0
        completed = None

        if task_id != 0:
            task = self._task_pool.get(task_id)
            if task is not None:
                if task["end_time"] >= self.t:               # valid: work on it
                    task["duration"] -= 1
                    if task["duration"] <= 0:                # task finished
                        time_taken = self.t - task["start_time"]
                        decay      = float(task["decay_rate"][0]) * time_taken
                        reward     = max(0.0, float(task["reward"][0]) - decay)
                        completed  = task
                        del self._task_pool[task_id]
                else:                                        # overdue: discard
                    del self._task_pool[task_id]

        # Record history
        category = (
            completed["category"]                if completed                  else
            self._task_pool[task_id]["category"] if task_id in self._task_pool else
            -1
        )
        self._agent_history[self.t]    = task_id
        self._task_history[self.t, 0]  = task_id
        self._task_history[self.t, 1]  = max(category, 0)

        # ── 4. Shift horizon left ──────────────────────────────────────────
        self._horizon[:-1] = self._horizon[1:]
        self._horizon[-1]  = 0

        # ── Advance clock ──────────────────────────────────────────────────
        self.t += 1
        self._prune_overdue()

        new_tasks = self._gen_tasks()
        self._register_tasks(new_tasks)

        terminated = self.t >= self.max_timesteps
        truncated  = False
        net_reward = reward - penalty

        return self._get_obs(new_tasks), net_reward, terminated, truncated, self._get_info(completed, penalty)

    # ── Rendering ──────────────────────────────────────────────────────────

    def render(self) -> None:
        preview   = self._horizon[:min(10, self.horizon_len)].tolist()
        pool_rows = sorted(self._task_pool.values(), key=lambda x: x["end_time"])
        print(f"\n── t={self.t:4d} ── pool={len(pool_rows):3d} tasks ─────────────────")
        print(f"  horizon (first 10): {preview}")
        for t in pool_rows:
            print(
                f"  id={t['id']:5d}  cat={t['category']}  "
                f"dur={t['duration']:3d}  deadline={t['end_time']:4d}  "
                f"base_reward={float(t['reward'][0]):.1f}  "
                f"decay={float(t['decay_rate'][0]):.3f}"
            )


# ── Policies ───────────────────────────────────────────────────────────────────

def build_fcfs_horizon(
    arrival_queue: deque,
    task_pool:     dict[int, dict],
    horizon_len:   int,
    current_t:     int,
) -> np.ndarray:
    """
    First-Come-First-Served horizon builder.

    Fills horizon slots left-to-right with tasks in arrival order.
    Each task occupies as many slots as its remaining duration.
    Tasks that are no longer in the pool (completed / expired) are skipped.
    """
    horizon = np.zeros(horizon_len, dtype=np.int32)
    slot    = 0
    for task_id in list(arrival_queue):
        if task_id not in task_pool:
            continue                          # already done or expired
        task      = task_pool[task_id]
        remaining = task["duration"]
        for _ in range(remaining):
            if slot >= horizon_len:
                return horizon
            horizon[slot] = task_id
            slot += 1
    return horizon


# ── Demo driver ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    categories = [
        TaskCategory(
            arrival_prob=0.10,
            duration_range=(5, 12), buffer_range=(3, 6),
            reward_range=(10.0, 20.0), decay_range=(0.1, 0.3),
        ),
        TaskCategory(
            arrival_prob=0.05,
            duration_range=(1,  6), buffer_range=(1, 2),
            reward_range=( 5.0, 10.0), decay_range=(0.05, 0.15),
        ),
        TaskCategory(
            arrival_prob=0.03,
            duration_range=(15, 30), buffer_range=(5, 10),
            reward_range=(50.0, 100.0), decay_range=(0.5, 1.0),
        ),
    ]

    env = SchedulerEnv(
        categories=categories,
        max_timesteps=200,
        horizon_len=15,
        penalty_coeff=0.5,
    )
    obs, _ = env.reset(seed=42)

    # FCFS state: queue holds task IDs in arrival order
    arrival_queue: deque[int] = deque()
    for task in obs["new_tasks"]:
        arrival_queue.append(task["id"])

    total_reward    = 0.0
    total_penalty   = 0.0
    completed_count = 0

    print("=" * 65)
    print(f"{'t':>4}  {'task_id':>7}  {'cat':>3}  {'reward':>8}  {'penalty':>8}")
    print("=" * 65)

    for _ in range(200):
        # Build FCFS horizon from current queue and pool
        horizon = build_fcfs_horizon(
            arrival_queue, env._task_pool, env.horizon_len, env.t
        )

        obs, net_reward, terminated, _, info = env.step(horizon)

        # Register newly arrived tasks (in arrival order)
        for task in obs["new_tasks"]:
            arrival_queue.append(task["id"])

        total_reward  += net_reward
        total_penalty += info["penalty"]

        if info["completed_task"]:
            ct = info["completed_task"]
            completed_count += 1
            print(
                f"{info['t']-1:4d}  {ct['id']:7d}  {ct['category']:3d}  "
                f"{net_reward + info['penalty']:8.2f}  -{info['penalty']:7.2f}"
            )

        if terminated:
            break

    print("=" * 65)
    print(f"\nCompleted : {completed_count} tasks")
    print(f"Gross rwrd: {total_reward + total_penalty:.2f}")
    print(f"Penalties : -{total_penalty:.2f}")
    print(f"Net reward: {total_reward:.2f}")