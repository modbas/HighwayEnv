import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import time
from matplotlib import pyplot as plt

import highway_env  # noqa: F401

TRAIN = True

if __name__ == "__main__":
    n_cpu = 24
    batch_size = 64
    env = make_vec_env("madtrack-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.9,
        verbose=2,
        tensorboard_log="madtrack_ppo/",
    )
    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(1e5))
        model.save("madtrack_ppo/model")
        del model

    # Run the algorithm
    env = gym.make("madtrack-v0", render_mode="rgb_array")
    model = PPO.load("madtrack_ppo/model", env=env)
    env = RecordVideo(
        env, video_folder="madtrack_ppo/videos", episode_trigger=lambda e: True
    )
    env.unwrapped.set_record_video_wrapper(env)

    for video in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
