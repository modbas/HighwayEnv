import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import time

import highway_env  # noqa: F401

TRAIN = False

if __name__ == "__main__":
    env = gym.make("madtrack-v0", render_mode="rgb_array")
    obs, info = env.reset()
    env.render()
    time.sleep(2)
    
    