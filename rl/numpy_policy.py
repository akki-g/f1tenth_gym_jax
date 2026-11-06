"""NumPy forward pass of the exported MLP actor (no JAX on the car).

Copied byte-for-byte into roboracer_ros2 (src/rl_policy/rl_policy/numpy_policy.py);
a test checks it matches the JAX actor's tanh(mean) within 1e-5.

policy.npz (written by rl/export_policy.py):
    W0, b0, ..., Wn, bn      actor MLP (Dense kernels are [in, out]): tanh on every
                             layer but the last (phase 1: n = 2)
    log_std                  state-independent log-std (unused at deployment)
    obs_mean, obs_std        obs_dim-long normalizer (identity on the scans)
    config                   JSON string: beam indices, v_cmd_max, steering limit,
                             policy rate, frame stack, norm clip, checkpoint hash, ...
"""

import json

import numpy as np


class NumpyPolicy:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            n_layers = sum(1 for k in data.files if k.startswith("W"))
            self.layers = [
                (data[f"W{i}"].astype(np.float64), data[f"b{i}"].astype(np.float64))
                for i in range(n_layers)
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
        """Pre-tanh action mean for one (obs_dim,) or a batch (N, obs_dim) of observations."""
        x = self.normalize(obs)
        for w, b in self.layers[:-1]:
            x = np.tanh(x @ w + b)
        w, b = self.layers[-1]
        return x @ w + b

    def __call__(self, obs):
        """Deterministic action (a_v, a_s) = tanh(mean), in (-1, 1)."""
        return np.tanh(self.mean(obs))
