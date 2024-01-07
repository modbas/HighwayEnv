import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import time
from matplotlib import pyplot as plt
import torch as th
import numpy as np

class OnnxablePolicy(th.nn.Module):
    def __init__(self, extractor, action_net, value_net):
        super().__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net

    def forward(self, observation):
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`
        action_hidden, value_hidden = self.extractor(observation)
        return self.action_net(action_hidden), self.value_net(value_hidden)

if __name__ == "__main__":
    env = gym.make("madtrack-v0", render_mode="rgb_array")
    model = PPO.load("madtrack_ppo/model", env=env, device="cpu")
    
    
    # ONNX export
    onnxable_model = OnnxablePolicy(
        model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net
    )
    observation_size = model.observation_space.shape
    dummy_input = th.randn(1, np.prod(observation_size))
    #obs, vectorenv_ = model.policy.obs_to_tensor(dummy_input)
    th.onnx.export(
        onnxable_model,
        dummy_input,
        "madtrack_ppo/model.onnx",
        opset_version=9,
        input_names=["input"],
    )

    # Torchscript trace
    jit_path = "madtrack_ppo/model.pt"
    #traced_module = th.jit.trace(onnxable_model.eval(), dummy_input)
    traced_module = th.jit.script(onnxable_model.eval(), dummy_input)
    frozen_module = th.jit.freeze(traced_module)
    frozen_module = th.jit.optimize_for_inference(frozen_module)
    th.jit.save(frozen_module, jit_path)


    env.close()
