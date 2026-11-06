"""Phase-1 PPO training (spec section 5), adapted from examples/train_ppo_example.py.

Changes from the example: multi-map phase-1 env (rl.maps.MultiMapEnv),
separate 256x256 tanh actor/critic on the 330-d observation, tanh-squashed
Gaussian with a state-independent log-std (init -0.5), running mean/std on
the 3 non-scan inputs, one jitted update per Python iteration (so we can
checkpoint, evaluate and resume), and GAE on the current step's done flag
(the example used the previous step's).

Outputs go to ``<runs root>/<YYYYMMDD>_<config hash>/``; starting again with
the same config resumes from that directory's latest checkpoint.

    python -m rl.train_ppo --num-envs 64 --time-limit 300          # CPU smoke
    python -m rl.train_ppo                                          # full config
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import time
import types
from typing import NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from flax import serialization  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402
from flax.traverse_util import flatten_dict, unflatten_dict  # noqa: E402
from safetensors.numpy import load_file, save_file  # noqa: E402

from . import car_params as cp  # noqa: E402
from . import maps as M  # noqa: E402
from .networks import (  # noqa: E402
    MODEL_KEYS,
    NORM_SLICE,
    Actor,
    CnnGru,
    Critic,
    Normalizer,
    gru_unroll,
    log_prob_from_pre_tanh,
    policy_distribution,
    sample_action,
)

# fields that do not change the experiment, left out of the config hash
_BOOKKEEPING = ("runs_root", "wandb_mode", "wandb_project", "time_limit", "map_dir", "fresh", "run_dir")


@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 0
    # PPO (spec section 5 table)
    num_envs: int = 2048
    num_steps: int = 256
    total_timesteps: int = 200_000_000
    num_minibatches: int = 32
    update_epochs: int = 4
    lr: float = 3e-4
    anneal_lr: bool = True
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.005  # phase 1c (phase 1: 0.001; its policy std collapsed 0.60 -> 0.03)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    optimizer: str = "adam"  # adam | adamw
    weight_decay: float = 0.0  # adamw only
    # network
    arch: str = "mlp"  # mlp | cnn_gru
    hidden_dim: int = 256  # mlp
    mlp_layers: int = 2  # mlp hidden layers
    log_std_init: float = -0.5
    gru_dim: int = 128  # cnn_gru
    proj_dim: int = 128  # cnn_gru: CNN features -> GRU input projection
    seq_len: int = 64  # cnn_gru: BPTT chunk; minibatches hold whole chunks
    gru_critic_detach: bool = False  # cnn_gru: value loss trains only the critic head, not the shared trunk
    # observation
    num_beams: int = cp.NUM_BEAMS  # 271 (phase 1: 109)
    frame_stack: int = 3  # mlp baselines 3 and 8; cnn_gru 1
    # env
    randomize: bool = True
    max_delay: int = cp.MAX_DELAY_STEPS  # action delay 0..max_delay policy steps per episode
    # phase 2 (1v1, cnn_gru): two cars, opponent pool, race reward, privileged critic
    race: bool = False
    init_from: str = ""  # phase-1c checkpoint: CNN, projection, GRU, actor head, normalizer (critic head is new)
    critic_warmup: int = 20  # first updates train only the critic head (actor learning rate 0)
    snapshot_every: int = 50  # updates between ego snapshots added to the opponent pool
    snapshot_pool: int = 8  # past ego snapshots kept, besides the starting model
    # episode ends beyond collisions (car_params; 0 turns a check off)
    wall_margin: float = cp.WALL_MARGIN
    wall_margin_noise: float = cp.WALL_MARGIN_NOISE
    wrong_way_steps: int = cp.WRONG_WAY_STEPS
    stall_steps: int = cp.STALL_STEPS
    train_maps: tuple = M.TRAIN_MAPS
    heldout_maps: tuple = M.HELDOUT_MAPS
    # checkpoints / evaluation
    checkpoint_every: int = 25
    eval_every: int = 50
    eval_episodes: int = 10
    eval_max_steps: int = 6000
    # bookkeeping
    runs_root: str = "runs"
    run_dir: str = ""
    map_dir: str = str(M.DEFAULT_MAP_DIR)
    wandb_mode: str = "offline"
    wandb_project: str = "roboracer-rl-phase1"
    time_limit: float = 0.0  # seconds; 0 = none. Stops cleanly with a checkpoint.
    fresh: bool = False  # ignore an existing run directory with the same hash

    def wrapper_kwargs(self) -> dict:
        return {"wall_margin": self.wall_margin, "wall_margin_noise": self.wall_margin_noise,
                "wrong_way_steps": self.wrong_way_steps, "stall_steps": self.stall_steps,
                "frame_stack": self.frame_stack, "max_delay": self.max_delay}

    def model_cfg(self) -> dict:
        from .race import PRIV_DIM

        cfg = {k: getattr(self, k) for k in MODEL_KEYS if k != "priv_dim"}
        cfg["priv_dim"] = PRIV_DIM if self.race else 0
        return cfg

    def validate(self):
        if self.optimizer not in ("adam", "adamw"):
            raise ValueError(f"optimizer must be adam or adamw, not {self.optimizer!r}")
        if self.mlp_layers < 1:
            raise ValueError("mlp_layers must be >= 1")
        if self.arch not in ("mlp", "cnn_gru"):
            raise ValueError(f"arch must be mlp or cnn_gru, not {self.arch!r}")
        if self.race and self.arch != "cnn_gru":
            raise ValueError("phase 2 (race) trains the cnn_gru")
        if self.init_from and self.arch != "cnn_gru":
            raise ValueError("init_from loads a cnn_gru checkpoint")
        if self.arch == "cnn_gru":
            if self.frame_stack != 1:
                raise ValueError("cnn_gru takes one frame (frame_stack 1); the GRU carries history")
            if self.num_steps % self.seq_len:
                raise ValueError("seq_len must divide num_steps")
            if (self.num_envs * self.num_steps // self.seq_len) % self.num_minibatches:
                raise ValueError("num_minibatches must divide the number of sequences")

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // (self.num_envs * self.num_steps)

    @property
    def minibatch_size(self) -> int:
        return self.num_envs * self.num_steps // self.num_minibatches

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["train_maps"], d["heldout_maps"] = list(self.train_maps), list(self.heldout_maps)
        return d

    def hash(self) -> str:
        d = {k: v for k, v in self.to_dict().items() if k not in _BOOKKEEPING}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


class Transition(NamedTuple):
    obs: jax.Array  # normalized, as the policy saw it
    pre_tanh: jax.Array
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    raw_norm_inputs: jax.Array  # raw speed / previous action, for the normalizer
    info: dict
    h_in: jax.Array  # cnn_gru: hidden state this step started from (after a reset); mlp: unused
    first: jax.Array  # this step starts an episode (the previous step was done)


class Runner(NamedTuple):
    actor: TrainState  # cnn_gru: the whole actor-critic (shared trunk)
    critic: TrainState | None  # cnn_gru: None
    normalizer: Normalizer
    env_states: tuple
    obs: jax.Array
    key: jax.Array
    update: jax.Array
    h: jax.Array  # cnn_gru: [num_envs, gru_dim] state after the last step; not checkpointed
    first: jax.Array  # [num_envs] the next step starts an episode
    opp_pool: dict | None = None  # race: stacked snapshot params / normalizers [pool, ...]; not checkpointed


INFO_KEYS = (
    "returned_episode", "episode_return", "episode_length", "episode_mean_speed",
    "episode_progress", "lap_complete", "collision", "timeout", "speed",
    "delay_k", "steer_offset", "param_mu", "param_m", "param_C_Sf", "param_C_Sr",
    "param_lf", "param_h", "param_I", "map_id",
    "contact", "wall_buffer", "wrong_way", "stall", "margin",
)
EPISODE_FIELDS = tuple(k for k in INFO_KEYS if k not in ("returned_episode", "speed"))
RACE_INFO_KEYS = ("car_contact", "won", "overtakes", "opp_type", "race_lead")


def _masked_mean(x, mask):
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1)


def _masked_std(x, mask):
    m = _masked_mean(x, mask)
    return jnp.sqrt(_masked_mean((x - m) ** 2, mask))


def compute_gae(reward, value, done, last_value, gamma, lam):
    """GAE over [T, N]. ``done[t]`` ends the episode at step t, so step t does
    not bootstrap from step t+1 (whose observation is already the reset one)."""

    def f(carry, t):
        adv, next_value = carry
        r, v, d = t
        delta = r + gamma * next_value * (1.0 - d) - v
        adv = delta + gamma * lam * (1.0 - d) * adv
        return (adv, v), adv

    _, adv = jax.lax.scan(f, (jnp.zeros_like(last_value), last_value), (reward, value, done.astype(jnp.float32)), reverse=True)
    return adv, adv + value


def _make_tx(config: Config):
    n_opt_steps = config.num_updates * config.update_epochs * config.num_minibatches
    lr = optax.linear_schedule(config.lr, 0.0, n_opt_steps) if config.anneal_lr else config.lr
    if config.optimizer == "adamw":
        opt = optax.adamw(lr, eps=1e-5, weight_decay=config.weight_decay)
    else:
        opt = optax.adam(lr, eps=1e-5)
    return optax.chain(optax.clip_by_global_norm(config.max_grad_norm), opt)


def _strip_weak_types(runner):
    # drop weak types (e.g. the env's jnp.full(n, 0)): update() outputs are
    # strongly typed, and a signature change would recompile it
    return jax.tree.map(lambda x: x.astype(x.dtype) if isinstance(x, jax.Array) else x, runner)


def _ppo_terms(config: Config, mean, log_std, value, t, adv, targets):
    """Clipped PPO policy and value losses (shared by both architectures)."""
    dist = policy_distribution(mean, log_std)
    logp = log_prob_from_pre_tanh(dist, t.pre_tanh)
    logratio = logp - t.log_prob
    ratio = jnp.exp(logratio)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * adv).mean()
    # entropy bonus on the pre-tanh Gaussian (the squashed one has no closed form)
    entropy = dist.distribution.entropy().mean()
    v_clipped = t.value + jnp.clip(value - t.value, -config.clip_eps, config.clip_eps)
    v_loss = 0.5 * jnp.maximum((value - targets) ** 2, (v_clipped - targets) ** 2).mean()
    total = pg - config.ent_coef * entropy + config.vf_coef * v_loss
    aux = {
        "policy_loss": pg,
        "value_loss": v_loss,
        "entropy": entropy,
        "approx_kl": ((ratio - 1) - logratio).mean(),
        "clipfrac": jnp.mean(jnp.abs(ratio - 1) > config.clip_eps),
    }
    return total, aux


def _update_metrics(aux, traj, normalizer, actor_params):
    info = traj.info
    fin = info["returned_episode"].astype(jnp.float32)
    metrics = {k: v.mean() for k, v in aux.items()}
    metrics.update(
        {
            "episodes_finished": fin.sum(),
            "episode_return": _masked_mean(info["episode_return"], fin),
            "episode_length": _masked_mean(info["episode_length"], fin),
            "completion_rate": _masked_mean(info["lap_complete"], fin),
            "collision_rate": _masked_mean(info["collision"], fin),
            "timeout_rate": _masked_mean(info["timeout"], fin),
            "contact_rate": _masked_mean(info["contact"], fin),
            "wall_buffer_rate": _masked_mean(info["wall_buffer"], fin),
            "wrong_way_rate": _masked_mean(info["wrong_way"], fin),
            "stall_rate": _masked_mean(info["stall"], fin),
            "episode_progress_laps": _masked_mean(info["episode_progress"], fin),
            "mean_speed": info["speed"].mean(),
            "mean_step_reward": traj.reward.mean(),
            "policy_std_speed": jnp.exp(actor_params["params"]["log_std"][0]),
            "policy_std_steer": jnp.exp(actor_params["params"]["log_std"][1]),
            "obs_norm_speed_mean": normalizer.mean[0],
            **{f"params/{k}_std": _masked_std(info[k], fin) for k in ("param_mu", "param_m", "delay_k", "steer_offset")},
            **{f"params/{k}_mean": _masked_mean(info[k], fin) for k in ("param_mu", "param_m", "delay_k")},
            "nan_count": sum(jnp.sum(~jnp.isfinite(x)) for x in jax.tree.leaves(actor_params)),
        }
    )
    episodes = {k: info[k] for k in EPISODE_FIELDS}
    if "won" in info:  # race: win rate and contact among opponent episodes
        opp = fin * (info["opp_type"] != 0)
        metrics.update({
            "race/win_rate": _masked_mean(info["won"], opp),
            "race/car_contact_rate": _masked_mean(info["car_contact"], opp),
            "race/overtakes": _masked_mean(info["overtakes"].astype(jnp.float32), opp),
            "race/opponent_episodes": opp.sum(),
        })
        episodes.update({k: info[k] for k in RACE_INFO_KEYS})
    episodes["finished"] = info["returned_episode"]
    return metrics, episodes


def make_trainer(config: Config, env: M.MultiMapEnv):
    """(init, jitted update, net, net-or-internals) for config.arch."""
    config.validate()
    if config.arch == "cnn_gru":
        return make_recurrent_trainer(config, env)
    return make_mlp_trainer(config, env)


def make_mlp_trainer(config: Config, env: M.MultiMapEnv):
    actor_net = Actor(config.hidden_dim, env.action_dim, config.log_std_init, config.mlp_layers)
    critic_net = Critic(config.hidden_dim, config.mlp_layers)

    def make_tx():
        return _make_tx(config)

    def init(key) -> Runner:
        k_actor, k_critic, k_env, key = jax.random.split(key, 4)
        dummy = jnp.zeros((1, env.obs_dim))
        # typed (not Python-int) step counters, so update() compiles only once
        step0 = jnp.zeros((), jnp.int32)
        actor = TrainState.create(apply_fn=actor_net.apply, params=actor_net.init(k_actor, dummy), tx=make_tx())
        critic = TrainState.create(apply_fn=critic_net.apply, params=critic_net.init(k_critic, dummy), tx=make_tx())
        obs, env_states = env.reset(k_env)
        n = obs.shape[0]
        runner = Runner(actor.replace(step=step0), critic.replace(step=step0), Normalizer.create(),
                        env_states, obs, key, step0, jnp.zeros((n, 1)), jnp.ones((n,), bool))
        return _strip_weak_types(runner)

    def env_step(carry, _):
        r: Runner = carry
        key, k_act, k_env = jax.random.split(r.key, 3)
        nobs = r.normalizer(r.obs)
        mean, log_std = actor_net.apply(r.actor.params, nobs)
        action, u, logp = sample_action(k_act, mean, log_std)
        value = critic_net.apply(r.critic.params, nobs)
        obs, env_states, reward, done, info = env.step(k_env, r.env_states, action)
        t = Transition(nobs, u, logp, value, reward, done, r.obs[:, NORM_SLICE], {k: info[k] for k in INFO_KEYS},
                       r.h, r.first)
        return r._replace(env_states=env_states, obs=obs, key=key, first=done), t

    def losses(actor_params, critic_params, batch):
        t, adv, targets = batch
        mean, log_std = actor_net.apply(actor_params, t.obs)
        value = critic_net.apply(critic_params, t.obs)
        return _ppo_terms(config, mean, log_std, value, t, adv, targets)

    grad_fn = jax.value_and_grad(losses, argnums=(0, 1), has_aux=True)

    def update(r: Runner):
        r, traj = jax.lax.scan(env_step, r, None, length=config.num_steps)
        last_value = critic_net.apply(r.critic.params, r.normalizer(r.obs))
        adv, targets = compute_gae(traj.reward, traj.value, traj.done, last_value, config.gamma, config.gae_lambda)

        flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), (traj._replace(info={}), adv, targets))

        def epoch(carry, _):
            actor, critic, key = carry
            key, k = jax.random.split(key)
            perm = jax.random.permutation(k, config.num_envs * config.num_steps)
            mbs = jax.tree.map(
                lambda x: x[perm].reshape((config.num_minibatches, -1) + x.shape[1:]), flat
            )

            def minibatch(carry, mb):
                actor, critic = carry
                (_, aux), (g_actor, g_critic) = grad_fn(actor.params, critic.params, mb)
                return (actor.apply_gradients(grads=g_actor), critic.apply_gradients(grads=g_critic)), aux

            (actor, critic), aux = jax.lax.scan(minibatch, (actor, critic), mbs)
            return (actor, critic, key), aux

        key, k_epochs = jax.random.split(r.key)
        (actor, critic, _), aux = jax.lax.scan(epoch, (r.actor, r.critic, k_epochs), None, length=config.update_epochs)
        normalizer = r.normalizer.update(traj.raw_norm_inputs)
        metrics, episodes = _update_metrics(aux, traj, normalizer, actor.params)
        new = Runner(actor, critic, normalizer, r.env_states, r.obs, key, r.update + 1, r.h, r.first, r.opp_pool)
        return new, metrics, episodes

    return init, jax.jit(update), actor_net, critic_net


def make_recurrent_trainer(config: Config, env: M.MultiMapEnv):
    """Recurrent PPO for cnn_gru (phase-1c spec, "Recurrent PPO changes").

    Rollout: h is carried per env and zeroed where the previous step ended an
    episode (``first``); each step stores the state it started from (h_in).
    Update: each env's num_steps rollout is cut into seq_len chunks; epochs
    permute whole chunks (not transitions) into num_minibatches minibatches;
    the loss reruns CNN+GRU over each chunk from its stored h_in, applying the
    same resets, and backpropagates through the chunk."""
    from .race import PRIV_DIM

    model = CnnGru(config.num_beams, config.gru_dim, config.proj_dim, action_dim=env.action_dim,
                   log_std_init=config.log_std_init, priv_dim=PRIV_DIM if config.race else 0)
    info_keys = INFO_KEYS + (RACE_INFO_KEYS if config.race else ())
    chunks = config.num_steps // config.seq_len
    n_seq = config.num_envs * chunks

    def init(key) -> Runner:
        k_net, k_env, key = jax.random.split(key, 3)
        obs, env_states = env.reset(k_env)
        n = obs.shape[0]
        h0 = jnp.zeros((n, config.gru_dim))
        params = model.init(k_net, h0[:1], jnp.zeros((1, env.obs_dim)))
        step0 = jnp.zeros((), jnp.int32)
        net = TrainState.create(apply_fn=model.apply, params=params, tx=_make_tx(config))
        normalizer = Normalizer.create()
        pool = init_pool(params, normalizer) if config.race else None
        runner = Runner(net.replace(step=step0), None, normalizer, env_states, obs, key, step0,
                        h0, jnp.ones((n,), bool), pool)
        return _strip_weak_types(runner)

    def init_pool(params, normalizer):
        """Opponent pool: slot 0 the starting model, slots 1..K past snapshots (start as copies)."""
        k = config.snapshot_pool + 1
        stack = lambda x: jnp.broadcast_to(x, (k,) + jnp.shape(x))  # noqa: E731
        return {"params": jax.tree.map(stack, params), "normalizer": jax.tree.map(stack, normalizer)}

    def policy_step(params, h, first, nobs):
        h_in = jnp.where(first[:, None], 0.0, h)
        h_out, mean, log_std, value = model.apply(params, h_in, nobs)
        return h_in, h_out, mean, log_std, value

    def env_step(carry, _):
        r: Runner = carry
        key, k_act, k_env = jax.random.split(r.key, 3)
        nobs = r.normalizer(r.obs)
        h_in, h_out, mean, log_std, value = policy_step(r.actor.params, r.h, r.first, nobs)
        action, u, logp = sample_action(k_act, mean, log_std)
        if config.race:
            obs, env_states, reward, done, info = env.step(k_env, r.env_states, action, r.opp_pool)
        else:
            obs, env_states, reward, done, info = env.step(k_env, r.env_states, action)
        t = Transition(nobs, u, logp, value, reward, done, r.obs[:, NORM_SLICE], {k: info[k] for k in info_keys},
                       h_in, r.first)
        return r._replace(env_states=env_states, obs=obs, key=key, h=h_out, first=done), t

    def to_sequences(x):
        """[T, N, ...] -> [N * chunks, seq_len, ...]; sequence i = env i // chunks,
        steps (i % chunks) * seq_len ... + seq_len - 1, in time order."""
        x = x.reshape((chunks, config.seq_len, config.num_envs) + x.shape[2:])
        x = jnp.moveaxis(x, 2, 0)
        return x.reshape((n_seq, config.seq_len) + x.shape[3:])

    def minibatches(seqs, key):
        """Permute whole sequences: leaves [n_seq, L, ...] -> [M, n_seq / M, L, ...]."""
        perm = jax.random.permutation(key, n_seq)
        return jax.tree.map(lambda x: x[perm].reshape((config.num_minibatches, -1) + x.shape[1:]), seqs)

    def recompute(params, t):
        """Rerun the policy over a batch of sequences [S, L, ...] from their stored
        start states. Returns (mean, log_std, value, h_in), batch-major."""
        tm = lambda x: jnp.swapaxes(x, 0, 1)  # noqa: E731
        mean, log_std, value, h_in, _ = gru_unroll(model, params, t.h_in[:, 0], tm(t.obs), tm(t.first),
                                                   detach_critic=config.gru_critic_detach)
        return tm(mean), log_std, tm(value), tm(h_in)

    def losses(params, batch):
        t, adv, targets = batch
        mean, log_std, value, _ = recompute(params, t)
        return _ppo_terms(config, mean, log_std, value, t, adv, targets)

    grad_fn = jax.value_and_grad(losses, has_aux=True)
    # critic warmup (phase 2): only the critic head learns while update < critic_warmup
    critic_only = None
    if config.race and config.critic_warmup > 0:
        def _critic_leaf(path, _):
            return any(getattr(p, "key", None) in ("critic_hidden", "critic_out") for p in path)

        critic_only = _critic_leaf

    def update(r: Runner):
        r, traj = jax.lax.scan(env_step, r, None, length=config.num_steps)
        *_, last_value = policy_step(r.actor.params, r.h, r.first, r.normalizer(r.obs))
        adv, targets = compute_gae(traj.reward, traj.value, traj.done, last_value, config.gamma, config.gae_lambda)
        seqs = jax.tree.map(to_sequences, (traj._replace(info={}), adv, targets))

        def epoch(carry, _):
            net, key = carry
            key, k = jax.random.split(key)

            def minibatch(net, mb):
                (_, aux), g = grad_fn(net.params, mb)
                if critic_only is not None:
                    warm = r.update < config.critic_warmup
                    g = jax.tree_util.tree_map_with_path(
                        lambda path, x: x if critic_only(path, x) else jnp.where(warm, 0.0, x), g)
                return net.apply_gradients(grads=g), aux

            net, aux = jax.lax.scan(minibatch, net, minibatches(seqs, k))
            return (net, key), aux

        key, k_epochs = jax.random.split(r.key)
        (net, _), aux = jax.lax.scan(epoch, (r.actor, k_epochs), None, length=config.update_epochs)
        normalizer = r.normalizer.update(traj.raw_norm_inputs)
        metrics, episodes = _update_metrics(aux, traj, normalizer, net.params)
        # GRU health: a state that stops moving, or saturates, is not being used
        metrics["gru_h_norm"] = jnp.linalg.norm(traj.h_in, axis=-1).mean()
        metrics["gru_h_saturated"] = jnp.mean(jnp.abs(traj.h_in) > 0.99)
        new = Runner(net, None, normalizer, r.env_states, r.obs, key, r.update + 1, r.h, r.first, r.opp_pool)
        return new, metrics, episodes

    internals = types.SimpleNamespace(init_pool=init_pool,model=model, env_step=env_step, policy_step=policy_step,
                                      to_sequences=to_sequences, minibatches=minibatches,
                                      recompute=recompute, losses=losses)
    return init, jax.jit(update), model, internals


# ---------------------------------------------------------------------------
# Checkpoints: safetensors with the config (and update index) in metadata
# ---------------------------------------------------------------------------
def _ckpt_tree(runner: Runner, key) -> dict:
    tree = {"actor": runner.actor, "normalizer": runner.normalizer, "key": key, "update": runner.update}
    if runner.critic is not None:  # cnn_gru keeps the critic head inside actor
        tree["critic"] = runner.critic
    return tree


def _state_dict(runner: Runner) -> dict:
    key = jax.random.key_data(runner.key) if jnp.issubdtype(runner.key.dtype, jax.dtypes.prng_key) else runner.key
    sd = serialization.to_state_dict(_ckpt_tree(runner, key))
    return {k: np.asarray(v) for k, v in flatten_dict(sd, sep="/").items()}


def save_checkpoint(path: pathlib.Path, runner: Runner, config: Config, extra: dict | None = None):
    path = pathlib.Path(path)
    meta = {"config": json.dumps(config.to_dict()), "config_hash": config.hash(),
            "update": str(int(runner.update)), "extra": json.dumps(extra or {})}
    tmp = path.with_suffix(".tmp")
    save_file(_state_dict(runner), str(tmp), metadata=meta)
    os.replace(tmp, path)  # atomic: a killed job never leaves a half-written file


def restore_runner(path, template: Runner) -> Runner:
    flat = load_file(str(path))
    tree = _ckpt_tree(template, template.key)
    base = flatten_dict(serialization.to_state_dict(tree), sep="/", keep_empty_nodes=True)
    merged = {k: (jnp.asarray(flat[k]) if k in flat else v) for k, v in base.items()}
    restored = serialization.from_state_dict(tree, unflatten_dict(merged, sep="/"))
    return template._replace(actor=restored["actor"], critic=restored.get("critic"),
                             normalizer=restored["normalizer"], key=restored["key"],
                             update=restored["update"])


def load_checkpoint(path) -> dict:
    """Actor params, normalizer and config from a checkpoint, no env needed."""
    from safetensors import safe_open

    path = pathlib.Path(path)
    flat = load_file(str(path))
    with safe_open(str(path), "np") as f:
        meta = f.metadata()
    config = json.loads(meta["config"])
    actor_flat = {k[len("actor/params/"):]: v for k, v in flat.items() if k.startswith("actor/params/")}
    actor_params = unflatten_dict(actor_flat, sep="/")
    normalizer = Normalizer(jnp.asarray(flat["normalizer/mean"]), jnp.asarray(flat["normalizer/var"]),
                            jnp.asarray(flat["normalizer/count"]))
    return {
        "actor_params": jax.tree.map(jnp.asarray, actor_params),
        "normalizer": normalizer,
        "config": config,
        "config_hash": meta.get("config_hash", ""),
        "update": int(meta["update"]),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "extra": json.loads(meta.get("extra", "{}")),
    }


# ---------------------------------------------------------------------------
# Phase 2: start from a phase-1c checkpoint; snapshot pool
# ---------------------------------------------------------------------------
def race_model(config: Config) -> CnnGru:
    """The race-format CnnGru (privileged critic), also used for snapshot opponents."""
    from .race import PRIV_DIM

    return CnnGru(config.num_beams, config.gru_dim, config.proj_dim, log_std_init=config.log_std_init,
                  priv_dim=PRIV_DIM)


def load_init_from(path, runner: Runner, init_pool=None):
    """Copy a phase-1c cnn_gru checkpoint's CNN, projection, GRU and actor head
    (and its normalizer) into a fresh race runner. The critic head stays newly
    initialized: its input gained the privileged block. Returns (runner, copied keys)."""
    ck = load_checkpoint(path)
    if ck["config"].get("arch") != "cnn_gru":
        raise ValueError(f"{path} is not a cnn_gru checkpoint")
    src = flatten_dict(ck["actor_params"], sep="/")
    dst = flatten_dict(runner.actor.params, sep="/")
    copied = [k for k, v in dst.items() if k in src and "critic_" not in k and tuple(src[k].shape) == tuple(v.shape)]
    missing = [k for k in dst if "critic_" not in k and k not in copied]
    if missing:
        raise ValueError(f"checkpoint lacks (or mis-shapes) actor/trunk parameters: {missing}")
    params = unflatten_dict({k: (jnp.asarray(src[k]) if k in copied else v) for k, v in dst.items()}, sep="/")
    net = runner.actor.replace(params=params, opt_state=runner.actor.tx.init(params))
    runner = runner._replace(actor=net, normalizer=ck["normalizer"])
    if init_pool is not None:
        runner = runner._replace(opp_pool=init_pool(params, ck["normalizer"]))
    return runner, copied


def add_snapshot(pool: dict, slot: int, params, normalizer) -> dict:
    return {"params": jax.tree.map(lambda p, x: p.at[slot].set(x), pool["params"], params),
            "normalizer": jax.tree.map(lambda p, x: p.at[slot].set(x), pool["normalizer"], normalizer)}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _resolve_run_dir(config: Config) -> pathlib.Path:
    if config.run_dir:
        return pathlib.Path(config.run_dir)
    root = pathlib.Path(config.runs_root)
    h = config.hash()
    if not config.fresh:
        existing = sorted(root.glob(f"*_{h}"))
        if existing:
            return existing[-1]
    return root / f"{datetime.date.today():%Y%m%d}_{h}"


def _write_episodes(fh, episodes, update, map_names):
    ep = jax.device_get(episodes)
    mask = np.asarray(ep.pop("finished"), bool)
    if not mask.any():
        return
    cols = {k: np.asarray(v)[mask] for k, v in ep.items()}
    for i in range(int(mask.sum())):
        row = {k: (float(v[i]) if v.dtype.kind == "f" else int(v[i])) for k, v in cols.items()}
        row["map"] = map_names[row.pop("map_id")]
        row["update"] = update
        fh.write(json.dumps(row) + "\n")


def train(config: Config) -> pathlib.Path:
    import wandb

    from .evaluate import make_evaluator

    config.validate()
    if config.num_updates < 1:
        raise ValueError("total_timesteps must cover one num_envs x num_steps rollout")
    if (config.num_envs * config.num_steps) % config.num_minibatches:
        raise ValueError("num_minibatches must divide num_envs * num_steps")

    M.use_map_dir(config.map_dir)
    run_dir = _resolve_run_dir(config)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    cfg_dict = {**config.to_dict(), "config_hash": config.hash(), "num_updates": config.num_updates,
                "maps": M.map_lists_config(),
                "car_params": {k: getattr(cp, k) for k in dir(cp) if k.isupper()},
                "jax_backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()]}
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2, default=str))
    manifest = pathlib.Path(config.map_dir) / "manifest.json"
    if manifest.exists():
        (run_dir / "maps_manifest.json").write_text(manifest.read_text())

    print(f"run dir: {run_dir}  ({config.num_updates} updates, backend {jax.default_backend()})", flush=True)
    t_build = time.time()
    race = dict(opponents=True, gru_dim=config.gru_dim, snapshot_model=race_model(config)) if config.race else None
    env = M.MultiMapEnv(config.train_maps, config.num_envs, randomize=config.randomize,
                        wrapper_kwargs=config.wrapper_kwargs(), race=race, num_beams=config.num_beams)
    init, update, _, internals = make_trainer(config, env)
    runner = jax.jit(init)(jax.random.PRNGKey(config.seed))
    latest = run_dir / "latest.safetensors"
    best_score = -np.inf
    if latest.exists():
        runner = restore_runner(latest, runner)
        best_info = load_checkpoint(latest)["extra"]
        best_score = best_info.get("best_score", -np.inf)
        print(f"resumed from {latest} at update {int(runner.update)}", flush=True)
    elif config.init_from:
        runner, copied = load_init_from(config.init_from, runner, internals.init_pool if config.race else None)
        print(f"initialized from {config.init_from}: {len(copied)} tensors (critic head new)", flush=True)
    if config.race and latest.exists():  # the pool is not checkpointed: restart it from the resumed policy
        runner = runner._replace(opp_pool=internals.init_pool(runner.actor.params, runner.normalizer))
    evaluate = make_evaluator(config.heldout_maps, config.eval_episodes, config.eval_max_steps,
                              wrapper_kwargs=config.wrapper_kwargs(),
                              randomize=False, model_cfg=config.model_cfg())

    wandb.init(project=config.wandb_project, name=run_dir.name, id=f"{run_dir.name}", resume="allow",
               dir=str(run_dir), mode=config.wandb_mode, config=cfg_dict)
    episodes_fh = open(run_dir / "episodes.jsonl", "a")
    metrics_fh = open(run_dir / "metrics.jsonl", "a")
    t_start = time.time()
    steps_per_update = config.num_envs * config.num_steps
    start_update = int(runner.update)
    print(f"setup {time.time() - t_build:.1f}s", flush=True)

    def checkpoint(tag_best=False):
        extra = {"best_score": float(best_score)}
        save_checkpoint(latest, runner, config, extra)
        u = int(runner.update)
        save_checkpoint(run_dir / f"ckpt_{u:05d}.safetensors", runner, config, extra)
        if tag_best:
            save_checkpoint(run_dir / "best.safetensors", runner, config, extra)

    try:
        while int(runner.update) < config.num_updates:
            t0 = time.time()
            runner, metrics, episodes = update(runner)
            if config.race and int(runner.update) % config.snapshot_every == 0:
                n_snap = int(runner.update) // config.snapshot_every
                slot = 1 + (n_snap - 1) % config.snapshot_pool  # slot 0 keeps the starting model
                runner = runner._replace(opp_pool=add_snapshot(runner.opp_pool, slot, runner.actor.params,
                                                               runner.normalizer))
            metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
            dt = time.time() - t0
            u = int(runner.update)
            metrics.update({"update": u, "env_steps": u * steps_per_update,
                            "sps": steps_per_update / dt, "update_seconds": dt,
                            "lr": float(config.lr * (1 - (u - 1) / config.num_updates)) if config.anneal_lr else config.lr})
            _write_episodes(episodes_fh, episodes, u, env.map_names)
            metrics_fh.write(json.dumps(metrics) + "\n")
            metrics_fh.flush()
            episodes_fh.flush()
            wandb.log(metrics, step=u)
            print(f"update {u:5d} | sps {metrics['sps']:8.0f} | ret {metrics['episode_return']:7.3f} | "
                  f"len {metrics['episode_length']:6.1f} | done {metrics['episodes_finished']:5.0f} | "
                  f"coll {metrics['collision_rate']:.2f} | prog {metrics['episode_progress_laps']:.3f} | "
                  f"v {metrics['mean_speed']:.2f} | std {metrics['policy_std_speed']:.3f}/{metrics['policy_std_steer']:.3f} | "
                  f"kl {metrics['approx_kl']:.4f}", flush=True)
            if not np.isfinite(metrics["episode_return"]) or metrics["nan_count"] > 0:
                raise FloatingPointError(f"NaN at update {u}")

            is_best = False
            out_of_time = config.time_limit and time.time() - t_start > config.time_limit
            if u % config.eval_every == 0 or (out_of_time and u > start_update):
                report = evaluate(runner.actor.params, runner.normalizer)
                report.update(update=u, env_steps=u * steps_per_update)
                (run_dir / "eval" / f"update_{u:05d}.json").write_text(json.dumps(report, indent=2))
                score = report["completion_rate"] + 1e-3 * report["mean_progress_laps"]
                is_best = score > best_score
                best_score = max(best_score, score)
                wandb.log({"eval/completion_rate": report["completion_rate"],
                           "eval/collision_rate": report["collision_rate"],
                           "eval/mean_progress_laps": report["mean_progress_laps"],
                           **{f"eval/{m}/completion": v["completion_rate"] for m, v in report["per_map"].items()}},
                          step=u)
                print(f"  eval: completion {report['completion_rate']:.2f} progress {report['mean_progress_laps']:.2f} "
                      f"({report['eval_seconds']:.0f}s){' new best' if is_best else ''}", flush=True)
            if u % config.checkpoint_every == 0 or is_best or out_of_time:
                checkpoint(tag_best=is_best)
            if out_of_time:
                print(f"time limit reached after {time.time() - t_start:.0f}s", flush=True)
                break
    finally:
        episodes_fh.close()
        metrics_fh.close()
        wandb.finish()
    if int(runner.update) >= config.num_updates:
        checkpoint()
    return run_dir


def parse_args(argv=None) -> Config:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            parser.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        elif f.name in ("train_maps", "heldout_maps"):
            parser.add_argument(flag, nargs="+", default=list(f.default))
        else:
            typ = {"int": lambda s: int(float(s)), "float": float, "str": str}.get(f.type, type(f.default))
            parser.add_argument(flag, type=typ, default=f.default)
    args = vars(parser.parse_args(argv))
    args["train_maps"], args["heldout_maps"] = tuple(args["train_maps"]), tuple(args["heldout_maps"])
    if isinstance(args["total_timesteps"], float):
        args["total_timesteps"] = int(args["total_timesteps"])
    return Config(**args)


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
