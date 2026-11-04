"""Actor, critic, tanh-squashed Gaussian policy and observation normalizer
(spec section 5). Shared by train_ppo.py, evaluate.py and export_policy.py."""

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
# the previous action. Scans are already in [0, 1].
NORM_SLICE = slice(ob.FRAME_STACK * ob.NUM_BEAMS, ob.OBS_DIM)  # 327:330
NORM_CLIP = 10.0


class MLP(nn.Module):
    """Two tanh hidden layers, orthogonal init (as in the PPO example)."""

    hidden_dim: int
    out_dim: int
    out_scale: float

    @nn.compact
    def __call__(self, x):
        for _ in range(2):
            x = nn.tanh(nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        return nn.Dense(self.out_dim, kernel_init=orthogonal(self.out_scale), bias_init=constant(0.0))(x)


class Actor(nn.Module):
    """Mean of the pre-tanh Gaussian, plus a state-independent learned log-std."""

    hidden_dim: int = 256
    action_dim: int = 2
    log_std_init: float = -0.5

    @nn.compact
    def __call__(self, x):
        mean = MLP(self.hidden_dim, self.action_dim, 0.01, name="mlp")(x)
        log_std = self.param("log_std", constant(self.log_std_init), (self.action_dim,))
        return mean, log_std


class Critic(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x):
        return MLP(self.hidden_dim, 1, 1.0, name="mlp")(x)[..., 0]


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
    def create(cls, dim: int = NORM_SLICE.stop - NORM_SLICE.start):
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

    def full_mean_std(self):
        """330-long mean/std with identity for the scans (exported to the car)."""
        mean = jnp.zeros(ob.OBS_DIM).at[NORM_SLICE].set(self.mean)
        std = jnp.ones(ob.OBS_DIM).at[NORM_SLICE].set(self.std())
        return mean, std

    def __call__(self, obs):
        normed = jnp.clip((obs[..., NORM_SLICE] - self.mean) / self.std(), -NORM_CLIP, NORM_CLIP)
        return obs.at[..., NORM_SLICE].set(normed)
