"""NumPy forward pass of an exported actor (no JAX on the car).

Copied byte-for-byte into roboracer_ros2 (src/rl_policy/rl_policy/numpy_policy.py);
tests check it matches the JAX actor's tanh(mean) within 1e-5.

policy.npz (written by rl/export_policy.py), config["arch"]:
    mlp (phase 1; files without "arch" are mlp)
        W0, b0, ..., Wn, bn      Dense kernels [in, out]: tanh on every layer but the last
    cnn_gru (phase 1c)
        conv{i}_kernel/_bias     Conv1d kernels [k, in, out], edge-padded, ReLU
        proj_kernel/_bias        flattened CNN features -> GRU input, ReLU
        gru_{ir,iz,in}_kernel/_bias, gru_{hr,hz}_kernel, gru_hn_kernel/_bias
                                 flax GRUCell: r = sig(x Wir + bir + h Whr),
                                 z = sig(x Wiz + biz + h Whz),
                                 n = tanh(x Win + bin + r * (h Whn + bhn)),
                                 h' = (1 - z) n + z h
        actor_hidden_*/actor_out_*   head: tanh hidden, linear out
    log_std                  state-independent log-std (unused at deployment)
    obs_mean, obs_std        obs_dim-long normalizer (identity on the scans)
    config                   JSON string: arch, beam indices, frame stack, v_cmd_max,
                             steering limit, policy rate, norm clip, checkpoint hash, ...

A cnn_gru policy is stateful: reset() zeroes the hidden state (start-up,
re-activation), and each call steps it once.
"""

import json

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _edge_pad(x, kernel, stride):
    """Pad the length axis (-2) for ceil(L / stride) VALID outputs, repeating edge values."""
    length = x.shape[-2]
    out = -(-length // stride)
    total = max((out - 1) * stride + kernel - length, 0)
    left = total // 2
    pad = [(0, 0)] * (x.ndim - 2) + [(left, total - left), (0, 0)]
    return np.pad(x, pad, mode="edge")


def _conv1d(x, kernel, bias, stride):
    """VALID Conv1d over the length axis: x [..., L, C_in], kernel [k, C_in, C_out].
    One BLAS matmul over flattened windows (einsum would not use BLAS here)."""
    k, c_in, c_out = kernel.shape
    win = np.lib.stride_tricks.sliding_window_view(x, k, axis=-2)[..., ::stride, :, :]  # [..., L_out, C_in, k]
    win = win.reshape(win.shape[:-2] + (c_in * k,))
    return win @ kernel.transpose(1, 0, 2).reshape(c_in * k, c_out) + bias


class NumpyPolicy:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files if k != "config"}
            self.config = json.loads(str(data["config"]))
        f64 = {k: v.astype(np.float64) for k, v in arrays.items()}
        self.arch = self.config.get("arch", "mlp")
        self.log_std = f64["log_std"]
        self.obs_mean = f64["obs_mean"]
        self.obs_std = f64["obs_std"]
        self.norm_clip = float(self.config["norm_clip"])
        self.obs_dim = int(self.config["obs_dim"])
        if self.arch == "mlp":
            n_layers = sum(1 for k in arrays if k.startswith("W"))
            self.layers = [(f64[f"W{i}"], f64[f"b{i}"]) for i in range(n_layers)]
            if self.layers[0][0].shape[0] != self.obs_dim:
                raise ValueError("policy input size does not match the config's obs_dim")
        elif self.arch == "cnn_gru":
            self.p = f64
            self.num_beams = int(self.config["num_beams"])
            self.conv_layers = [tuple(c) for c in self.config["conv_layers"]]  # (features, kernel, stride)
            self.gru_dim = int(self.config["gru_dim"])
            if self.obs_dim != self.num_beams + 3:
                raise ValueError("cnn_gru policy expects one scan + speed + previous action")
            self.h = np.zeros(self.gru_dim)
        else:
            raise ValueError(f"unknown policy arch {self.arch!r}")

    def normalize(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        return np.clip((obs - self.obs_mean) / self.obs_std, -self.norm_clip, self.norm_clip)

    def reset(self):
        """Zero the recurrent state (no-op for an MLP)."""
        if self.arch == "cnn_gru":
            self.h = np.zeros(self.gru_dim)

    def forward(self, obs, h=None):
        """Pre-tanh action mean for one (obs_dim,) or a batch (N, obs_dim) of
        observations. cnn_gru: also takes and returns the hidden state
        (h=None: the policy's own state, not updated here)."""
        x = self.normalize(obs)
        if self.arch == "mlp":
            for w, b in self.layers[:-1]:
                x = np.tanh(x @ w + b)
            w, b = self.layers[-1]
            return x @ w + b, None
        p = self.p
        h = self.h if h is None else np.asarray(h, dtype=np.float64)
        z = x[..., : self.num_beams, None]
        for i, (_, k, s) in enumerate(self.conv_layers):
            z = np.maximum(_conv1d(_edge_pad(z, k, s), p[f"conv{i}_kernel"], p[f"conv{i}_bias"], s), 0.0)
        z = z.reshape(z.shape[:-2] + (-1,))
        z = np.maximum(z @ p["proj_kernel"] + p["proj_bias"], 0.0)
        g = np.concatenate([z, x[..., self.num_beams :]], axis=-1)  # + speed, previous action
        r = _sigmoid(g @ p["gru_ir_kernel"] + p["gru_ir_bias"] + h @ p["gru_hr_kernel"])
        u = _sigmoid(g @ p["gru_iz_kernel"] + p["gru_iz_bias"] + h @ p["gru_hz_kernel"])
        n = np.tanh(g @ p["gru_in_kernel"] + p["gru_in_bias"] + r * (h @ p["gru_hn_kernel"] + p["gru_hn_bias"]))
        h_new = (1.0 - u) * n + u * h
        a = np.tanh(h_new @ p["actor_hidden_kernel"] + p["actor_hidden_bias"])
        return a @ p["actor_out_kernel"] + p["actor_out_bias"], h_new

    def mean(self, obs):
        """Pre-tanh action mean (stateless for an MLP; does not advance a GRU's state)."""
        return self.forward(obs)[0]

    def __call__(self, obs):
        """Deterministic action (a_v, a_s) = tanh(mean), in (-1, 1). A cnn_gru
        policy advances its hidden state by one step."""
        mean, h = self.forward(obs)
        if self.arch == "cnn_gru":
            self.h = h
        return np.tanh(mean)
