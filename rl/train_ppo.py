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
    NORM_SLICE,
    Actor,
    Critic,
    Normalizer,
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
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # network
    hidden_dim: int = 256
    log_std_init: float = -0.5
    # env
    randomize: bool = True
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


class Runner(NamedTuple):
    actor: TrainState
    critic: TrainState
    normalizer: Normalizer
    env_states: tuple
    obs: jax.Array
    key: jax.Array
    update: jax.Array


INFO_KEYS = (
    "returned_episode", "episode_return", "episode_length", "episode_mean_speed",
    "episode_progress", "lap_complete", "collision", "timeout", "speed",
    "delay_k", "steer_offset", "param_mu", "param_m", "param_C_Sf", "param_C_Sr",
    "param_lf", "param_h", "param_I", "map_id",
)
EPISODE_FIELDS = tuple(k for k in INFO_KEYS if k not in ("returned_episode", "speed"))


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


def make_trainer(config: Config, env: M.MultiMapEnv):
    actor_net = Actor(config.hidden_dim, env.action_dim, config.log_std_init)
    critic_net = Critic(config.hidden_dim)
    n_opt_steps = config.num_updates * config.update_epochs * config.num_minibatches

    def make_tx():
        lr = optax.linear_schedule(config.lr, 0.0, n_opt_steps) if config.anneal_lr else config.lr
        return optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(lr, eps=1e-5))

    def init(key) -> Runner:
        k_actor, k_critic, k_env, key = jax.random.split(key, 4)
        dummy = jnp.zeros((1, env.obs_dim))
        # typed (not Python-int) step counters, so update() compiles only once
        step0 = jnp.zeros((), jnp.int32)
        actor = TrainState.create(apply_fn=actor_net.apply, params=actor_net.init(k_actor, dummy), tx=make_tx())
        critic = TrainState.create(apply_fn=critic_net.apply, params=critic_net.init(k_critic, dummy), tx=make_tx())
        obs, env_states = env.reset(k_env)
        runner = Runner(actor.replace(step=step0), critic.replace(step=step0), Normalizer.create(),
                        env_states, obs, key, step0)
        # drop weak types (e.g. the env's jnp.full(n, 0)): update() outputs are
        # strongly typed, and a signature change would recompile it
        return jax.tree.map(lambda x: x.astype(x.dtype) if isinstance(x, jax.Array) else x, runner)

    def env_step(carry, _):
        r: Runner = carry
        key, k_act, k_env = jax.random.split(r.key, 3)
        nobs = r.normalizer(r.obs)
        mean, log_std = actor_net.apply(r.actor.params, nobs)
        action, u, logp = sample_action(k_act, mean, log_std)
        value = critic_net.apply(r.critic.params, nobs)
        obs, env_states, reward, done, info = env.step(k_env, r.env_states, action)
        t = Transition(nobs, u, logp, value, reward, done, r.obs[:, NORM_SLICE], {k: info[k] for k in INFO_KEYS})
        return r._replace(env_states=env_states, obs=obs, key=key), t

    def losses(actor_params, critic_params, batch):
        t, adv, targets = batch
        mean, log_std = actor_net.apply(actor_params, t.obs)
        dist = policy_distribution(mean, log_std)
        logp = log_prob_from_pre_tanh(dist, t.pre_tanh)
        logratio = logp - t.log_prob
        ratio = jnp.exp(logratio)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * adv).mean()
        # entropy bonus on the pre-tanh Gaussian (the squashed one has no closed form)
        entropy = dist.distribution.entropy().mean()
        value = critic_net.apply(critic_params, t.obs)
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
                "episode_progress_laps": _masked_mean(info["episode_progress"], fin),
                "mean_speed": info["speed"].mean(),
                "mean_step_reward": traj.reward.mean(),
                "policy_std_speed": jnp.exp(actor.params["params"]["log_std"][0]),
                "policy_std_steer": jnp.exp(actor.params["params"]["log_std"][1]),
                "obs_norm_speed_mean": normalizer.mean[0],
                **{f"params/{k}_std": _masked_std(info[k], fin) for k in ("param_mu", "param_m", "delay_k", "steer_offset")},
                **{f"params/{k}_mean": _masked_mean(info[k], fin) for k in ("param_mu", "param_m", "delay_k")},
                "nan_count": sum(jnp.sum(~jnp.isfinite(x)) for x in jax.tree.leaves(actor.params)),
            }
        )
        episodes = {k: info[k] for k in EPISODE_FIELDS}
        episodes["finished"] = info["returned_episode"]
        new = Runner(actor, critic, normalizer, r.env_states, r.obs, key, r.update + 1)
        return new, metrics, episodes

    return init, jax.jit(update), actor_net, critic_net


# ---------------------------------------------------------------------------
# Checkpoints: safetensors with the config (and update index) in metadata
# ---------------------------------------------------------------------------
def _state_dict(runner: Runner) -> dict:
    sd = serialization.to_state_dict(
        {"actor": runner.actor, "critic": runner.critic, "normalizer": runner.normalizer,
         "key": jax.random.key_data(runner.key) if jnp.issubdtype(runner.key.dtype, jax.dtypes.prng_key) else runner.key,
         "update": runner.update}
    )
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
    base = flatten_dict(serialization.to_state_dict(
        {"actor": template.actor, "critic": template.critic, "normalizer": template.normalizer,
         "key": template.key, "update": template.update}), sep="/", keep_empty_nodes=True)
    merged = {k: (jnp.asarray(flat[k]) if k in flat else v) for k, v in base.items()}
    sd = unflatten_dict(merged, sep="/")
    restored = serialization.from_state_dict(
        {"actor": template.actor, "critic": template.critic, "normalizer": template.normalizer,
         "key": template.key, "update": template.update}, sd)
    return template._replace(actor=restored["actor"], critic=restored["critic"],
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
    env = M.MultiMapEnv(config.train_maps, config.num_envs, randomize=config.randomize)
    init, update, _, _ = make_trainer(config, env)
    runner = jax.jit(init)(jax.random.PRNGKey(config.seed))
    latest = run_dir / "latest.safetensors"
    best_score = -np.inf
    if latest.exists():
        runner = restore_runner(latest, runner)
        best_info = load_checkpoint(latest)["extra"]
        best_score = best_info.get("best_score", -np.inf)
        print(f"resumed from {latest} at update {int(runner.update)}", flush=True)
    evaluate = make_evaluator(config.heldout_maps, config.eval_episodes, config.eval_max_steps,
                              randomize=False, hidden_dim=config.hidden_dim)

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
