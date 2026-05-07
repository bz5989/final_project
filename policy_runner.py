import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

# ============================================================
# Part 1: Policy Network (Discrete Actions)
# ============================================================

class Policy(nn.Module):
    """A simple MLP policy that outputs a Categorical distribution.

    Architecture:
        obs (obs_dim) -> Linear(hidden_size) -> ReLU
                      -> Linear(hidden_size) -> ReLU
                      -> Linear(act_dim)  [logits]

    The forward pass should return a torch.distributions.Categorical
    constructed from the logits (NOT softmax probabilities).
    """

    def __init__(self, obs_dim=8, act_dim=4, hidden_size=64):
        super().__init__()
        # TODO: Define the network layers (e.g., using nn.Sequential).
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )

    def forward(self, obs):
        """
        Args:
            obs: Observation tensor, shape [batch, obs_dim] or [obs_dim]

        Returns:
            dist: torch.distributions.Categorical over actions
        """
        # TODO: Forward pass through the network, return Categorical(logits=...).
        return Categorical(logits=self.network(obs))

    def get_action(self, obs):
        """Sample an action and return (action, log_prob).

        Args:
            obs: numpy array of shape [obs_dim]

        Returns:
            action: int
            log_prob: scalar tensor (keep in the computation graph)
        """
        # TODO: Convert obs to a float32 tensor, get the distribution from
        # forward(), sample an action, compute its log_prob, and return both.
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        dist = self.forward(obs_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def get_log_prob(self, obs, actions):
        """Compute log probabilities of given actions under the current policy.

        Args:
            obs: tensor of shape [batch, obs_dim]
            actions: tensor of shape [batch]

        Returns:
            log_probs: tensor of shape [batch]
        """
        # TODO: Forward pass, then compute log_prob of the given actions.
        return self.forward(obs).log_prob(actions)


# ============================================================
# Part 1: Rollout Collection
# ============================================================

def collect_rollout(env, policy, max_steps=1000):
    """Collect a single trajectory by running the policy in the environment.

    Args:
        env: gymnasium environment
        policy: Policy network
        max_steps: Maximum steps per episode

    Returns:
        dict with keys:
            'observations': list of numpy arrays, shape [obs_dim] each
            'actions': list of ints
            'rewards': list of floats
            'log_probs': list of scalar tensors
            'total_reward': float (sum of rewards)
    """
    # TODO: Reset the environment, run the policy until done or max_steps,
    # collecting observations, actions, rewards, and log_probs at each step.
    obs = env.reset()[0]
    action = policy.get_action(obs)[0]
    steps = 0
    ret = {}
    ret['observations'] = []
    ret['actions'] = []
    ret['rewards'] = []
    ret['log_probs'] = []
    ret['total_reward'] = 0.0
    while steps < max_steps:
        obs, reward, done, truncated, _ = env.step(action)
        if done or truncated:
            break
        ret['observations'].append(obs)
        action, log_prob = policy.get_action(obs)
        ret['actions'].append(action)
        ret['rewards'].append(reward)
        ret['log_probs'].append(log_prob)
        ret['total_reward'] += reward
        steps += 1
    return ret

def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}.

    Args:
        rewards: list of floats, length T
        gamma: discount factor

    Returns:
        returns: numpy array of shape [T], where returns[t] = G_t
    """
    # TODO: Compute discounted returns for each timestep.
    # Hint: work backwards from the last timestep.
    t = len(rewards)
    returns = np.zeros(t)
    returns[-1] = rewards[-1]
    for i in range(t - 2, -1, -1):
        returns[i] = rewards[i] + gamma * returns[i + 1]

    return returns


# ============================================================
# Part 1: REINFORCE
# ============================================================

def reinforce_update(policy, optimizer, log_probs, returns, baseline=None):
    """Perform one REINFORCE gradient update.

    The REINFORCE loss is:
        L = -1/T * sum_t log pi(a_t|s_t) * (G_t - b)

    where b is an optional baseline (if None, use b=0).

    Args:
        policy: Policy network
        optimizer: torch optimizer
        log_probs: list of scalar tensors, length T
        returns: numpy array of shape [T]
        baseline: float or None. If None, no baseline is subtracted.

    Returns:
        loss: float (the scalar loss value, for logging)
    """
    # TODO: Compute the REINFORCE policy gradient loss, backprop, and step.
    # 1. Convert returns to a tensor
    returns_tensor = torch.tensor(returns, dtype=torch.float32)
    # 2. If baseline is not None, subtract it from the returns
    if baseline is not None:
        returns_tensor -= baseline
    # 3. Compute loss
    loss = 0.0
    for log_prob, G in zip(log_probs, returns_tensor):
        loss += -log_prob * G
    loss /= len(log_probs)
    # 4. optimizer.zero_grad(), loss.backward(), optimizer.step()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 5. Return loss.item()
    return loss.item()


def train_reinforce(env, policy, optimizer, total_steps=500000, gamma=0.99,
                    use_baseline=True, eval_episode_interval=50, verbose=True):
    """Train a policy using REINFORCE.

    Run episodes until at least total_steps environment steps have been
    collected.  If use_baseline is True, subtract the mean of the returns
    as a baseline (i.e., baseline = mean(G_t) for each episode).

    Args:
        env: gymnasium environment
        policy: Policy network
        optimizer: torch optimizer
        total_steps: Total environment steps budget
        gamma: Discount factor
        use_baseline: Whether to use a mean-return baseline
        eval_episode_interval: Evaluate every this many episodes.
        verbose: If True, print progress at each eval point.
            Set to False for parallel multi-seed runs.

    Returns:
        training_rewards: list of total rewards per training episode
        eval_rewards: list of (episode_count, mean_eval_reward) tuples
        losses: list of loss values per episode
    """
    training_rewards = []
    eval_rewards = []
    losses = []

    steps_so_far = 0
    ep = 0
    next_eval_ep = eval_episode_interval

    while steps_so_far < total_steps:
        # TODO: Fill in the training step.
        #   1. Collect a rollout using collect_rollout
        rollout = collect_rollout(env, policy)
        #   2. Compute returns using compute_returns
        returns = compute_returns(rollout['rewards'], gamma)
        #   3. If use_baseline, set baseline = mean of the returns; else None
        baseline = np.mean(returns) if use_baseline else None
        #   4. Call reinforce_update
        loss = reinforce_update(policy, optimizer, rollout['log_probs'], returns, baseline)

        # --- Bookkeeping (do not modify) ---
        ep_len = len(rollout['rewards'])
        steps_so_far += ep_len
        ep += 1
        training_rewards.append(rollout['total_reward'])
        losses.append(loss)

        if ep >= next_eval_ep:
            mean_rew, _ = evaluate_policy(env, policy)
            eval_rewards.append((ep, mean_rew))
            next_eval_ep += eval_episode_interval
            if verbose:
                print(f"  Episode {ep:5d} ({steps_so_far:>8,} steps) | "
                      f"Eval: {mean_rew:7.1f}")

    return training_rewards, eval_rewards, losses


# ============================================================
# Part 2: PPO
# ============================================================

class ValueNetwork(nn.Module):
    """Value function approximator V(s).

    Architecture:
        obs (obs_dim) -> Linear(hidden_size) -> ReLU
                      -> Linear(hidden_size) -> ReLU
                      -> Linear(1)
    """

    def __init__(self, obs_dim=8, hidden_size=64):
        super().__init__()
        # TODO: Define the value network layers.
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs):
        """
        Args:
            obs: tensor of shape [batch, obs_dim] or [obs_dim]

        Returns:
            value: tensor of shape [batch, 1] or [1]
        """
        # TODO: Forward pass, return value estimate.
        return self.network(obs)


def collect_batch(env, policy, num_steps=2048, gamma=0.99, lam=0.95,
                  value_net=None):
    """Collect a batch of experience for PPO and compute GAE advantages.

    Run the policy in the environment for num_steps total steps (which may
    span multiple episodes).  When an episode ends, reset and continue.

    Must handle BOTH discrete and continuous action spaces:
      - Use isinstance(env.action_space, gym.spaces.Discrete) to detect.
      - For discrete: actions are ints, tensor dtype = torch.long.
      - For continuous: actions are floats, tensor dtype = torch.float32,
        and you must wrap the action as np.array([action]) for env.step().

    Generalized Advantage Estimation (GAE):
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = sum_{l=0}^{T-t-1} (gamma * lam)^l * delta_{t+l}

    After computing advantages, normalize them to have mean 0 and std 1.
    Returns are computed as: returns = advantages + values.

    Args:
        env: gymnasium environment
        policy: Policy network (discrete or continuous)
        num_steps: Number of environment steps to collect
        gamma: Discount factor
        lam: GAE lambda parameter
        value_net: ValueNetwork for computing value estimates

    Returns:
        batch: dict with keys:
            'observations': tensor [num_steps, obs_dim]
            'actions': tensor [num_steps] (long for discrete, float for continuous)
            'log_probs': tensor [num_steps] (detached from computation graph)
            'returns': tensor [num_steps]
            'advantages': tensor [num_steps]
            'episode_rewards': list of total rewards for completed episodes
    """
    # TODO: Collect experience and compute GAE advantages.
    # Steps:
    #   1. Detect action space type: is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    #   2. Initialize lists for obs, actions, rewards, dones, log_probs, values.
    obs_list = []
    action_list = []
    reward_list = []
    done_list = []
    log_prob_list = []
    value_list = []
    episode_rewards = []
    episode_reward = 0.0
    #   3. Reset env, get initial obs.
    obs, _ = env.reset()
    #   4. For num_steps iterations:
    for _ in range(num_steps):
        #      a. Get distribution from policy, sample action, compute log_prob (detach).
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        dist = policy.forward(obs_tensor)
        # if discrete, action = int; if continuous, action = float
        action = dist.sample()
        log_prob = dist.log_prob(action).detach()
        #      b. Get value from value_net (detach).
        value = value_net(obs_tensor).detach().item()
        #      c. Step the environment:
        #         - Discrete: env.step(action.item())
        #         - Continuous: env.step(np.array([action.item()]))
        if is_discrete:
            next_obs, reward, done, truncated, _ = env.step(action.item())
        else:
            next_obs, reward, done, truncated, _ = env.step(np.array([action.item()]))
        #      d. Store everything.
        obs_list.append(obs)
        action_list.append(action.item())
        reward_list.append(reward)
        episode_reward += reward
        done_list.append(done)
        log_prob_list.append(log_prob)
        value_list.append(value)
        obs = next_obs
        #      e. If episode ends, reset env and track the episode reward.
        if done or truncated:
            episode_rewards.append(episode_reward)
            obs, _ = env.reset()
            episode_reward = 0.0
    #   5. Compute the bootstrap value V(s_{num_steps}) for the last state.
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
    value = value_net(obs_tensor).detach().item()
    value_list.append(value)
    # episode_rewards.append(episode_reward)

    rewards = np.array(reward_list)
    values = np.array(value_list)
    dones = np.array(done_list)

    #   6. Compute GAE advantages using the delta/advantage recursion above.
    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
    advantages = np.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    #   7. returns = advantages + values
    returns = advantages + values[:-1]
    #   8. Normalize advantages: (adv - mean) / (std + 1e-8)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #   9. Convert everything to tensors (use torch.long for discrete actions,
    #      torch.float32 for continuous actions) and return.
    batch = {
        'observations': torch.tensor(np.array(obs_list, dtype=np.float32), dtype=torch.float32),
        'actions': torch.tensor(action_list, dtype=torch.long if is_discrete else torch.float32),
        'log_probs': torch.tensor(np.array(log_prob_list), dtype=torch.float32),
        'returns': torch.tensor(np.array(returns), dtype=torch.float32),
        'advantages': torch.tensor(np.array(advantages), dtype=torch.float32),
        'episode_rewards': episode_rewards,
    }
    return batch


def ppo_update(policy, value_net, optimizer, batch, clip_epsilon=0.2,
               value_coef=0.5, entropy_coef=0.01, num_epochs=4,
               minibatch_size=64):
    """Perform PPO update with clipped surrogate objective.

    The PPO-Clip objective for each sample is:
        ratio = exp(log_pi_new(a|s) - log_pi_old(a|s))
        L_clip = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)

    Total loss per minibatch:
        L = -mean(L_clip) + value_coef * MSE(V(s), returns) - entropy_coef * H

    where H is the mean entropy of the policy distribution.

    Args:
        policy: Policy network
        value_net: Value network
        optimizer: torch optimizer (for both policy and value_net parameters)
        batch: dict from collect_batch
        clip_epsilon: PPO clipping parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy bonus coefficient
        num_epochs: Number of passes over the full batch
        minibatch_size: Minibatch size for each gradient step

    Returns:
        dict with mean 'policy_loss', 'value_loss', 'entropy', 'clip_fraction'
    """
    # TODO: Implement the PPO clipped update.
    policy_losses = 0.0
    value_losses = 0.0
    entropies = 0.0
    clip_fractions = 0.0
    tot_div = num_epochs * len(batch['observations']) // minibatch_size
    # For each epoch:
    for _ in range(num_epochs):
        #   1. Generate a random permutation of indices [0, num_steps).
        indices = torch.randperm(len(batch['observations']))
        #   2. Split into minibatches of size minibatch_size.
        #   3. For each minibatch:
        for start in range(0, len(batch['observations']), minibatch_size):
            end = min(start + minibatch_size, len(batch['observations']))
            mb_indices = indices[start:end]
            #      a. Extract obs, actions, old_log_probs, returns, advantages.
            obs = batch['observations'][mb_indices]
            actions = batch['actions'][mb_indices]
            log_probs = batch['log_probs'][mb_indices]
            returns = batch['returns'][mb_indices]
            advantages = batch['advantages'][mb_indices]
            #      b. Compute new log probs via policy.get_log_prob(obs, actions).
            new_log_probs = policy.get_log_prob(obs, actions)
            dist = policy.forward(obs)
            #      c. Compute entropy of the policy distribution.
            entropy = dist.entropy().mean()
            #      d. ratio = exp(new_log_probs - old_log_probs)
            ratio = torch.exp(new_log_probs - log_probs)
            # print(ratio)
            #      e. surr1 = ratio * advantages
            #      f. surr2 = clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantages
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            #      g. policy_loss = -mean(min(surr1, surr2))
            #      h. value_loss = MSE(value_net(obs).squeeze(), returns)
            policy_loss = -torch.mean(torch.min(surr1, surr2))
            value_loss = nn.MSELoss()(value_net(obs).squeeze(-1), returns)
            #      i. loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            #      j. optimizer.zero_grad(), loss.backward(), optimizer.step()   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
             #      k. Track clip_fraction = mean(|ratio - 1| > clip_epsilon)
            clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_epsilon).float()).item()
            policy_losses += policy_loss.item()
            value_losses += value_loss.item()
            entropies += entropy.item()
            clip_fractions += clip_fraction
    #   4. Return mean metrics across all minibatches.
    return {
        'policy_loss': policy_losses / tot_div,
        'value_loss': value_losses / tot_div,
        'entropy': entropies / tot_div,
        'clip_fraction': clip_fractions / tot_div,
    }


def train_ppo(env, policy, value_net, optimizer, total_steps=500000,
              steps_per_iter=2048, gamma=0.99, lam=0.95, clip_epsilon=0.2,
              num_epochs=4, minibatch_size=64, eval_episode_interval=50,
              verbose=True):
    """Train a policy using PPO.

    Compute num_iterations = total_steps // steps_per_iter, then run that
    many iterations of collect_batch + ppo_update.

    Args:
        env: gymnasium environment
        policy: Policy network
        value_net: ValueNetwork
        optimizer: torch optimizer
        total_steps: Total environment steps budget
        steps_per_iter: Steps collected per iteration
        gamma, lam: GAE parameters
        clip_epsilon: PPO clipping parameter
        num_epochs: PPO update epochs per iteration
        minibatch_size: Minibatch size
        eval_episode_interval: Evaluate every this many completed episodes.
        verbose: If True, print progress at each eval point.
            Set to False for parallel multi-seed runs.

    Returns:
        training_rewards: list of mean episode rewards per iteration
        eval_rewards: list of (episode_count, mean_eval_reward) tuples
        metrics: dict of lists with keys 'policy_loss', 'value_loss',
                 'entropy', 'clip_fraction'
    """
    training_rewards = []
    eval_rewards = []
    metrics = {'policy_loss': [], 'value_loss': [], 'entropy': [],
               'clip_fraction': []}

    num_iterations = total_steps // steps_per_iter
    total_episodes = 0
    next_eval_ep = eval_episode_interval

    for it in range(num_iterations):
        # TODO: Fill in the training step.
        #   1. Call collect_batch to gather experience.
        #   2. Call ppo_update to update policy and value_net.
        batch = collect_batch(env, policy, num_steps=steps_per_iter,
                                          gamma=gamma, lam=lam, value_net=value_net)
        update_info = ppo_update(policy, value_net, optimizer, batch,
                                 clip_epsilon=clip_epsilon, num_epochs=num_epochs,
                                 minibatch_size=minibatch_size)
        # print(batch['episode_rewards'])
        # --- Bookkeeping (do not modify) ---
        ep_rews = batch['episode_rewards']
        mean_rew = np.mean(ep_rews) if ep_rews else 0.0
        training_rewards.append(mean_rew)
        total_episodes += len(ep_rews)

        for key in metrics:
            metrics[key].append(update_info[key])

        if total_episodes >= next_eval_ep:
            eval_mean, _ = evaluate_policy(env, policy)
            eval_rewards.append((total_episodes, eval_mean))
            next_eval_ep += eval_episode_interval
            if verbose:
                step_count = (it + 1) * steps_per_iter
                print(f"  Episode {total_episodes:5d} ({step_count:>8,} steps) | "
                      f"Train: {mean_rew:7.1f} | "
                      f"Eval: {eval_mean:7.1f} | "
                      f"Clip: {update_info['clip_fraction']:.3f}")

    return training_rewards, eval_rewards, metrics


# ============================================================
# Part 3: Gaussian Policy (Continuous Actions)
# ============================================================

class GaussianPolicy(nn.Module):
    """A Gaussian MLP policy for continuous action spaces.

    Architecture:
        obs (obs_dim) -> Linear(hidden_size) -> ReLU
                      -> Linear(hidden_size) -> ReLU
                      -> Linear(act_dim)  [action mean]

    Also has a learnable log_std parameter of shape [act_dim].
    The forward pass returns a torch.distributions.Normal distribution.

    For act_dim=1, squeeze the output so the distribution has the same
    batch shape as the observation (this makes log_prob return scalars
    for single observations, matching the discrete policy interface).
    """

    def __init__(self, obs_dim, act_dim, hidden_size=64,
                 action_low=-2.0, action_high=2.0):
        super().__init__()
        # TODO: Note, make a decision of whether you're going to use a fixed variance or a learnable variance, which is more stable? You don't have to report this, but you'll have to experiment to get things working.
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
        # self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)
        self.act_dim = act_dim
        self.action_low = action_low
        self.action_high = action_high

    def forward(self, obs):
        """
        Args:
            obs: Observation tensor, shape [batch, obs_dim] or [obs_dim]

        Returns:
            dist: torch.distributions.Normal over actions
        """
        # TODO:
        loc = self.network(obs)
        log_std = torch.clamp(self.log_std, min=-20, max=2)
        scale = torch.exp(log_std)
        return Normal(loc=loc, scale=scale)

    def get_action(self, obs):
        """Sample an action and return (action, log_prob).

        Args:
            obs: numpy array of shape [obs_dim]

        Returns:
            action: numpy array of shape [act_dim] (clamped to action bounds)
            log_prob: scalar tensor
        """
        # TODO:
        # 1. Convert obs to float32 tensor
        # 2. Get Normal distribution from forward()
        # 3. Sample an action
        # 4. Compute log_prob (scalar for 1D actions)
        # 5. Clamp the action to [action_low, action_high]
        # 6. Return action as numpy array of shape [act_dim], and log_prob
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        dist = self.forward(obs_tensor)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.squeeze(-1)
        action_clamped = torch.clamp(action, self.action_low, self.action_high)
        return action_clamped.detach().numpy(), log_prob

    def get_log_prob(self, obs, actions):
        """Compute log probabilities of given actions.

        Args:
            obs: tensor of shape [batch, obs_dim]
            actions: tensor of shape [batch]

        Returns:
            log_probs: tensor of shape [batch]
        """
        # TODO: Forward pass, compute log_prob of given actions.
        log_prob = self.forward(obs).log_prob(actions)
        log_prob = log_prob.squeeze(-1)
        return log_prob

# ============================================================
# Multi-seed experiment worker (must be at module level)
# ============================================================

def _run_experiment(args):
    """Worker function for parallel multi-seed experiments."""
    seed, config = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(config['env_name'])

    # Build policy
    if config['policy_class'] == 'Policy':
        policy = Policy(**config['policy_kwargs'])
    else:
        policy = GaussianPolicy(**config['policy_kwargs'])

    if config['algo'] == 'reinforce':
        optimizer = optim.Adam(policy.parameters(), lr=config['lr'])
        train_rew, eval_rew, _ = train_reinforce(
            env, policy, optimizer, verbose=False, **config['train_kwargs'])
        metrics = None
    else:  # ppo
        value_net = ValueNetwork(**config['value_kwargs'])
        optimizer = optim.Adam(
            list(policy.parameters()) + list(value_net.parameters()),
            lr=config['lr'])
        train_rew, eval_rew, metrics = train_ppo(
            env, policy, value_net, optimizer, verbose=False,
            **config['train_kwargs'])

    final_eval = eval_rew[-1][1] if eval_rew else float('-inf')
    policy_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

    env.close()
    return {
        'eval_rewards': eval_rew,
        'training_rewards': train_rew,
        'final_eval': final_eval,
        'policy_state': policy_state,
        'metrics': metrics,
    }


# ============================================================
# Main -- runs everything and generates plots
# ============================================================

if __name__ == "__main__":
    import os
    import argparse
    from multiprocessing import Pool

    parser = argparse.ArgumentParser(
        description="HW2: Policy Gradient Methods")
    parser.add_argument('--num-seeds', type=int, default=5,
                        help='Number of random seeds per experiment')
    parser.add_argument('--num-workers', type=int, default=5,
                        help='Number of parallel workers')
    cli_args = parser.parse_args()

    plot_dir = os.path.join(os.path.dirname(__file__), "plots")
    gif_dir = os.path.join(os.path.dirname(__file__), "gifs")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)

    NUM_SEEDS = cli_args.num_seeds
    NUM_WORKERS = cli_args.num_workers
    seeds = list(range(NUM_SEEDS))

    TOTAL_STEPS = 300_000
    PENDULUM_STEPS = 200_000

    # ----------------------------------------------------------
    # Experiment configurations
    # ----------------------------------------------------------
    experiments = {
        'reinforce_lunar': {
            'algo': 'reinforce',
            'env_name': 'LunarLander-v3',
            'policy_class': 'Policy',
            'policy_kwargs': {'obs_dim': 8, 'act_dim': 4, 'hidden_size': 64},
            'lr': 1e-3,
            'train_kwargs': {
                'total_steps': TOTAL_STEPS, 'gamma': 0.99,
                'use_baseline': True,
            },
        },
        'ppo_lunar': {
            'algo': 'ppo',
            'env_name': 'LunarLander-v3',
            'policy_class': 'Policy',
            'policy_kwargs': {'obs_dim': 8, 'act_dim': 4, 'hidden_size': 64},
            'value_kwargs': {'obs_dim': 8, 'hidden_size': 64},
            'lr': 5e-5,
            'train_kwargs': {
                'total_steps': TOTAL_STEPS, 'steps_per_iter': 2048,
                'gamma': 0.99, 'lam': 0.95, 'clip_epsilon': 0.2,
                'num_epochs': 3,
            },
        },
        'reinforce_pendulum': {
            'algo': 'reinforce',
            'env_name': 'Pendulum-v1',
            'policy_class': 'GaussianPolicy',
            'policy_kwargs': {'obs_dim': 3, 'act_dim': 1, 'hidden_size': 64,
                              'action_low': -2.0, 'action_high': 2.0},
            'lr': 1e-3,
            'train_kwargs': {
                'total_steps': PENDULUM_STEPS, 'gamma': 0.99,
                'use_baseline': True,
            },
        },
        'ppo_pendulum': {
            'algo': 'ppo',
            'env_name': 'Pendulum-v1',
            'policy_class': 'GaussianPolicy',
            'policy_kwargs': {'obs_dim': 3, 'act_dim': 1, 'hidden_size': 64,
                              'action_low': -2.0, 'action_high': 2.0},
            'value_kwargs': {'obs_dim': 3, 'hidden_size': 64},
            'lr': 3e-5,
            'train_kwargs': {
                'total_steps': PENDULUM_STEPS, 'steps_per_iter': 2048,
                'gamma': 0.99, 'lam': 0.95, 'clip_epsilon': 0.2,
                'num_epochs': 3, 'minibatch_size': 64,
            },
        },
    }

    # ----------------------------------------------------------
    # Run all experiments (parallel across seeds)
    # ----------------------------------------------------------
    all_results = {}
    for name, config in experiments.items():
        print(f"\n{'='*60}")
        print(f"  {name}  ({NUM_SEEDS} seeds, {NUM_WORKERS} workers)")
        print(f"{'='*60}")
        tasks = [(seed, config) for seed in seeds]
        with Pool(NUM_WORKERS) as pool:
            results = list(pool.map(_run_experiment, tasks))
        all_results[name] = results
        evals = [r['final_eval'] for r in results]
        print(f"  Final eval: {np.mean(evals):.1f} +/- {np.std(evals):.1f}")

def compute_policy_metrics(env, policy, num_episodes=100, max_steps=1000):
    """Compute detailed evaluation metrics for a trained policy.

    Useful for diagnosing failure modes: high variance indicates unreliable
    landing, low success rate with moderate mean reward suggests the policy
    can slow its descent but crashes on landing.

    Args:
        env: gymnasium environment
        policy: Policy network
        num_episodes: Number of evaluation episodes
        max_steps: Max steps per episode

    Returns:
        dict with keys:
            'mean_reward': Mean total reward
            'std_reward': Std of total rewards (measures policy reliability)
            'min_reward': Worst episode reward
            'max_reward': Best episode reward
            'success_rate': Fraction of episodes with reward > 200
            'crash_rate': Fraction of episodes with reward < -100
            'mean_length': Mean episode length
            'rewards': list of all episode rewards
            'lengths': list of all episode lengths
    """
    rewards = []
    lengths = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=2000 + ep)
        total_reward = 0.0
        steps = 0
        for _ in range(max_steps):
            with torch.no_grad():
                action, _ = policy.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        rewards.append(total_reward)
        lengths.append(steps)

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'success_rate': np.mean([r > 200 for r in rewards]),
        'crash_rate': np.mean([r < -100 for r in rewards]),
        'mean_length': np.mean(lengths),
        'rewards': rewards,
        'lengths': lengths,
    }

def evaluate_policy(env, policy, num_episodes=20, max_steps=1000):
    """Evaluate a policy by running deterministic rollouts.

    Args:
        env: gymnasium environment
        policy: Policy network
        num_episodes: Number of evaluation episodes
        max_steps: Max steps per episode

    Returns:
        mean_reward: Mean total reward across episodes
        rewards: list of total rewards per episode
    """
    rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=1000 + ep)
        total_reward = 0.0
        for _ in range(max_steps):
            with torch.no_grad():
                action, _ = policy.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards), rewards