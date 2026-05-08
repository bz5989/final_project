import gymnasium as gym
from stable_baselines3 import A2C

env = gym.make("CartPole-v1", render_mode="rgb_array")
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_enc = model.get_env()
obs = vec_enc.reset()

for i in range(1_000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_enc.step(action)
    vec_enc.render("human")