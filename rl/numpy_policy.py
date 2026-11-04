"""NumPy forward pass of the exported phase-1 actor (no JAX on the car).

Copied byte-for-byte into roboracer_ros2 (src/rl_policy/rl_policy/numpy_policy.py);
a test checks it matches the JAX actor's tanh(mean) within 1e-5.

policy.npz (written by rl/export_policy.py):
    W0, b0, W1, b1, W2, b2   actor MLP (Dense kernels are [in, out]), tanh hidden
    log_std                  state-independent log-std (unused at deployment)
    obs_mean, obs_std        330-long normalizer (identity on the scans)
    config                   JSON string: beam indices, v_cmd_max, steering limit,
                             policy rate, frame stack, norm clip, checkpoint hash, ...
"""

import json

import numpy as np


class NumpyPolicy:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.layers = [
                (data[f"W{i}"].astype(np.float64), data[f"b{i}"].astype(np.float64))
                for i in range(3)
            ]
            self.log_std = data["log_std"].astype(np.float64)
            self.obs_mean = data["obs_mean"].astype(np.float64)
            self.obs_std = data["obs_std"].astype(np.float64)
            self.config = json.loads(str(data["config"]))
        self.norm_clip = float(self.config["norm_clip"])
        self.obs_dim = int(self.config["obs_dim"])
        if self.layers[0][0].shape[0] != self.obs_dim:
            raise ValueError("policy input size does not match the config's obs_dim")

    def normalize(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        return np.clip((obs - self.obs_mean) / self.obs_std, -self.norm_clip, self.norm_clip)

    def mean(self, obs):
        """Pre-tanh action mean for one (330,) or a batch (N, 330) of observations."""
        x = self.normalize(obs)
        (w0, b0), (w1, b1), (w2, b2) = self.layers
        x = np.tanh(x @ w0 + b0)
        x = np.tanh(x @ w1 + b1)
        return x @ w2 + b2

    def __call__(self, obs):
        """Deterministic action (a_v, a_s) = tanh(mean), in (-1, 1)."""
        return np.tanh(self.mean(obs))
