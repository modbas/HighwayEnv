import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import time
from matplotlib import pyplot as pl
import highway_env  # noqa: F401
import torch as th
import numpy as np

if __name__ == "__main__":
    # Run the deployed algorithm
    env = gym.make("madtrack-v0", render_mode="rgb_array")
    model = th.jit.load("madtrack_ppo/model.pt")
    env = RecordVideo(
        env, video_folder="madtrack_ppo/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)


    for video in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Convert to tensor
            tensor_obs = th.tensor(np.reshape(obs, (1, 144)), dtype=th.float32)
            # Predict
            action = model(tensor_obs)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
