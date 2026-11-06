"""Actor, critic, tanh-squashed Gaussian policy and observation normalizer
(spec section 5), and the phase-1c CNN+GRU actor-critic. Shared by
train_ppo.py, evaluate.py and export_policy.py.

Architectures (``arch``):
    mlp      phase-1 256x256 tanh actor and separate critic on the stacked obs
    cnn_gru  edge-padded 1D CNN on the scan -> Dense 128 -> GRU 128 (speed and
             previous action join at the GRU input) -> actor and critic heads
"""

from __future__ import annotations

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.linen.initializers import constant, orthogonal

from . import obs as ob

# The observation entries that get running mean/std normalization: speed and
# the previous action, the last 3 of any layout. Scans are already in [0, 1].
NORM_SLICE = slice(-ob.NUM_NON_SCAN, None)
NORM_CLIP = 10.0


class MLP(nn.Module):
    """``layers`` tanh hidden layers (phase 1: 2), orthogonal init (as in the PPO example)."""

    hidden_dim: int
    out_dim: int
    out_scale: float
    layers: int = 2

    @nn.compact
    def __call__(self, x):
        for _ in range(self.layers):
            x = nn.tanh(nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        return nn.Dense(self.out_dim, kernel_init=orthogonal(self.out_scale), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    """Mean of the pre-tanh Gaussian, plus a state-independent learned log-std."""

    hidden_dim: int = 256
    action_dim: int = 2
    log_std_init: float = -0.5
    layers: int = 2

    @nn.compact
    def __call__(self, x):
        mean = MLP(self.hidden_dim, self.action_dim, 0.01, self.layers, name="mlp")(x)
        log_std = self.param("log_std", constant(self.log_std_init), (self.action_dim,))
        return mean, log_std


class Critic(nn.Module):
    hidden_dim: int = 256
    layers: int = 2

    @nn.compact
    def __call__(self, x):
        return MLP(self.hidden_dim, 1, 1.0, self.layers, name="mlp")(x)[..., 0]


def policy_distribution(mean, log_std) -> distrax.Transformed:
    """Diagonal Gaussian squashed by tanh (distrax Transformed + Tanh)."""
    base = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std) * jnp.ones_like(mean))
    return distrax.Transformed(base, distrax.Block(distrax.Tanh(), ndims=1))


def log_prob_from_pre_tanh(dist: distrax.Transformed, u):
    """log pi(tanh(u)), evaluated from the pre-tanh sample u.

    Equals dist.log_prob(tanh(u)) but avoids atanh(+-1) when the action saturates.
    """
    return dist.distribution.log_prob(u) - dist.bijector.forward_log_det_jacobian(u)


def sample_action(key, mean, log_std):
    """Returns (action in (-1, 1), pre-tanh sample, log-prob)."""
    dist = policy_distribution(mean, log_std)
    u = dist.distribution.sample(seed=key)
    return jnp.tanh(u), u, log_prob_from_pre_tanh(dist, u)


def deterministic_action(mean):
    """Evaluation and deployment action: tanh(mean)."""
    return jnp.tanh(mean)


@struct.dataclass
class Normalizer:
    """Running mean/std of the non-scan observation entries."""

    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @classmethod
    def create(cls, dim: int = ob.NUM_NON_SCAN):
        return cls(jnp.zeros(dim), jnp.ones(dim), jnp.array(1e-4, jnp.float32))

    def update(self, x):
        """Merge a batch [..., dim] (parallel Welford)."""
        x = x.reshape(-1, x.shape[-1])
        b_mean, b_var, b_count = x.mean(0), x.var(0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        mean = self.mean + delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / tot
        return Normalizer(mean, m2 / tot, tot)

    def std(self):
        return jnp.sqrt(self.var + 1e-8)

    def full_mean_std(self, obs_dim: int):
        """obs_dim-long mean/std with identity for the scans (exported to the car)."""
        mean = jnp.zeros(obs_dim).at[NORM_SLICE].set(self.mean)
        std = jnp.ones(obs_dim).at[NORM_SLICE].set(self.std())
        return mean, std

    def __call__(self, obs):
        normed = jnp.clip((obs[..., NORM_SLICE] - self.mean) / self.std(), -NORM_CLIP, NORM_CLIP)
        return obs.at[..., NORM_SLICE].set(normed)


# ---------------------------------------------------------------------------
# Phase 1c: CNN + GRU actor-critic
# ---------------------------------------------------------------------------
# (features, kernel, stride) per Conv1d; 271 beams -> 136 -> 68 -> 34 -> 17
CONV_LAYERS = ((16, 5, 2), (32, 5, 2), (32, 3, 2), (32, 3, 2))


def edge_pad(x, kernel: int, stride: int):
    """Pad the length axis (-2) so a VALID conv gives ceil(L / stride) outputs,
    repeating the edge values. A normalized 0 is an obstacle at 0 m, so zero
    padding would put a fake wall at both ends of the 270 deg field."""
    length = x.shape[-2]
    out = -(-length // stride)
    total = max((out - 1) * stride + kernel - length, 0)
    left = total // 2
    pad = [(0, 0)] * (x.ndim - 2) + [(left, total - left), (0, 0)]
    return jnp.pad(x, pad, mode="edge")


class CnnGru(nn.Module):
    """Shared CNN + projection + GRU trunk, actor and critic heads.

    One policy step: ``h' = GRU(h, [proj(CNN(scan)), speed, prev action])``,
    then ``mean, log_std = actor(h')`` and ``value = critic(h')``. The
    observation must be a single frame (frame stack 1)."""

    num_beams: int = ob.NUM_BEAMS
    gru_dim: int = 128
    proj_dim: int = 128
    head_dim: int = 64
    action_dim: int = 2
    log_std_init: float = -0.5

    def setup(self):
        hid = orthogonal(np.sqrt(2))
        self.convs = [nn.Conv(f, (k,), strides=(st,), padding="VALID", kernel_init=hid, bias_init=constant(0.0))
                      for f, k, st in CONV_LAYERS]
        self.proj = nn.Dense(self.proj_dim, kernel_init=hid, bias_init=constant(0.0))
        self.gru = nn.GRUCell(features=self.gru_dim, kernel_init=orthogonal(1.0), recurrent_kernel_init=orthogonal(1.0))
        self.actor_hidden = nn.Dense(self.head_dim, kernel_init=hid, bias_init=constant(0.0))
        self.actor_out = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.log_std = self.param("log_std", constant(self.log_std_init), (self.action_dim,))
        self.critic_hidden = nn.Dense(self.head_dim, kernel_init=hid, bias_init=constant(0.0))
        self.critic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def conv1(self, scan):
        """First conv layer on an edge-padded [..., num_beams] scan (for tests)."""
        _, k, st = CONV_LAYERS[0]
        return self.convs[0](edge_pad(scan[..., None], k, st))

    def encode(self, obs):
        """[..., obs_dim] -> GRU input [..., proj_dim + 3]."""
        x = obs[..., : self.num_beams, None]
        for conv, (_, k, st) in zip(self.convs, CONV_LAYERS):
            x = nn.relu(conv(edge_pad(x, k, st)))
        x = nn.relu(self.proj(x.reshape(x.shape[:-2] + (-1,))))
        return jnp.concatenate([x, obs[..., self.num_beams:]], axis=-1)

    def step(self, h, x):
        h, _ = self.gru(h, x)
        return h

    def heads(self, h):
        mean = self.actor_out(nn.tanh(self.actor_hidden(h)))
        value = self.critic_out(nn.tanh(self.critic_hidden(h)))[..., 0]
        return mean, self.log_std, value

    def __call__(self, h, obs):
        h = self.step(h, self.encode(obs))
        mean, log_std, value = self.heads(h)
        return h, mean, log_std, value


def gru_unroll(model: CnnGru, params, h0, obs_seq, first_seq, detach_critic: bool = False):
    """Run the recurrent policy over [T, B, obs] from h0 [B, H].

    ``first_seq[t]`` marks an episode start at step t: the hidden state is
    zeroed before the step, exactly as in the rollout. Returns
    (mean [T,B,A], log_std, value [T,B], h_in [T,B,H], h_out [T,B,H]); h_in[t]
    is the state step t started from (after any reset). ``detach_critic``
    stops the value loss from training the shared trunk (only the critic head
    learns from it)."""
    x = model.apply(params, obs_seq, method=CnnGru.encode)

    def f(h, inp):
        x_t, first_t = inp
        h_in = jnp.where(first_t[:, None], 0.0, h)
        h = model.apply(params, h_in, x_t, method=CnnGru.step)
        return h, (h_in, h)

    _, (h_in, h_out) = jax.lax.scan(f, h0, (x, first_seq))
    mean, log_std, value = model.apply(params, h_out, method=CnnGru.heads)
    if detach_critic:
        _, _, value = model.apply(params, jax.lax.stop_gradient(h_out), method=CnnGru.heads)
    return mean, log_std, value, h_in, h_out


# ---------------------------------------------------------------------------
# Model description shared by training, evaluation and export
# ---------------------------------------------------------------------------
MODEL_KEYS = ("arch", "num_beams", "frame_stack", "hidden_dim", "mlp_layers", "gru_dim", "proj_dim", "log_std_init")


def model_cfg_from(config: dict) -> dict:
    """Model fields of a run config; phase-1 configs lack them (109-beam MLP)."""
    from . import car_params as cp

    out = {"arch": "mlp", "num_beams": cp.LEGACY_LAYOUT["num_beams"], "frame_stack": cp.LEGACY_LAYOUT["frame_stack"],
           "hidden_dim": 256, "mlp_layers": 2, "gru_dim": 128, "proj_dim": 128, "log_std_init": -0.5}
    out.update({k: config[k] for k in MODEL_KEYS if k in config})
    return out


def make_policy(model_cfg: dict, action_dim: int = 2):
    """Deterministic acting for evaluation: returns (init_carry(n), act) with
    ``act(params, normalizer, obs, carry, first) -> (tanh(mean), carry)``.
    ``first`` marks envs whose episode starts at this step (GRU state reset)."""
    if model_cfg["arch"] == "mlp":
        actor = Actor(model_cfg["hidden_dim"], action_dim, model_cfg["log_std_init"], model_cfg["mlp_layers"])

        def act(params, normalizer, obs, carry, first):
            mean, _ = actor.apply(params, normalizer(obs))
            return deterministic_action(mean), carry

        return (lambda n: jnp.zeros((n, 1))), act
    if model_cfg["arch"] == "cnn_gru":
        model = CnnGru(model_cfg["num_beams"], model_cfg["gru_dim"], model_cfg["proj_dim"], action_dim=action_dim,
                       log_std_init=model_cfg["log_std_init"])

        def act(params, normalizer, obs, carry, first):
            h_in = jnp.where(first[:, None], 0.0, carry)
            h, mean, _, _ = model.apply(params, h_in, normalizer(obs))
            return deterministic_action(mean), h

        return (lambda n: jnp.zeros((n, model_cfg["gru_dim"]))), act
    raise ValueError(f"unknown arch {model_cfg['arch']!r}")
