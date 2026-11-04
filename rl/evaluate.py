"""Held-out evaluation (spec section 5): deterministic actions (tanh(mean)),
fixed start points, a JSON report.

Default setting: nominal vehicle parameters, no noise, delay or steering
offset; ``--randomize`` turns all of them on (the robustness gate).

An episode *completes* when the car finishes one full lap without a
collision. Training episodes run 2 laps / 90 s, but most racetracks are
260-550 m long, so 2 laps in 90 s would need more than v_cmd_max on them;
evaluation therefore uses 1 lap and a longer time limit (``max_steps``).

    python -m rl.evaluate --checkpoint runs/<run>/best.safetensors [--randomize]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from . import car_params as cp
from . import maps as M
from .networks import Actor, Normalizer, deterministic_action

DEFAULT_EVAL_MAX_STEPS = 6000  # 240 s at 25 Hz


def make_evaluator(
    map_names=M.HELDOUT_MAPS,
    episodes_per_map: int = 10,
    max_steps: int = DEFAULT_EVAL_MAX_STEPS,
    randomize: bool = False,
    seed: int = 12345,
    hidden_dim: int = 256,
):
    """Returns evaluate(actor_params, normalizer) -> report dict."""
    from f1tenth_gym_jax.envs import F110Env

    from .domain_rand import make_param
    from .wrappers import Phase1Wrapper

    actor = Actor(hidden_dim=hidden_dim)
    wrappers = [
        Phase1Wrapper(
            F110Env(1, make_param(m, randomize=randomize, max_steps=max_steps)),
            randomize,
            max_laps=1,
        )
        for m in map_names
    ]

    def run_map(w, actor_params, normalizer, key):
        n = episodes_per_map
        starts = jnp.arange(n) * (w.track_length / n)  # fixed, evenly spaced
        k_reset, k_run = jax.random.split(key)
        obs, states = jax.vmap(w.reset_at)(jax.random.split(k_reset, n), starts)
        zeros = jnp.zeros(n)
        init = dict(
            t=0,
            obs=obs,
            states=states,
            key=k_run,
            finished=jnp.zeros(n, bool),
            lap=jnp.zeros(n, bool),
            collision=jnp.zeros(n, bool),
            timeout=jnp.zeros(n, bool),
            length=zeros,
            speed=zeros,
            progress=zeros,
        )

        def cond(c):
            return (c["t"] < max_steps + 1) & ~jnp.all(c["finished"])

        def body(c):
            key, k = jax.random.split(c["key"])
            mean, _ = actor.apply(actor_params, normalizer(c["obs"]))
            action = deterministic_action(mean)
            obs, states, _, done, info = jax.vmap(w.step)(jax.random.split(k, n), c["states"], action)
            first = done & ~c["finished"]

            def record(old, new):
                return jnp.where(first, new, old)

            return dict(
                t=c["t"] + 1,
                obs=obs,
                states=states,
                key=key,
                finished=c["finished"] | done,
                lap=record(c["lap"], info["lap_complete"]),
                collision=record(c["collision"], info["collision"]),
                timeout=record(c["timeout"], info["timeout"]),
                length=record(c["length"], info["episode_length"].astype(jnp.float32)),
                speed=record(c["speed"], info["episode_mean_speed"]),
                progress=record(c["progress"], info["episode_progress"]),
            )

        out = jax.lax.while_loop(cond, body, init)
        return {k: out[k] for k in ("finished", "lap", "collision", "timeout", "length", "speed", "progress")}

    @jax.jit
    def run_all(actor_params, normalizer):
        keys = jax.random.split(jax.random.PRNGKey(seed), len(wrappers))
        return [run_map(w, actor_params, normalizer, k) for w, k in zip(wrappers, keys)]

    def evaluate(actor_params, normalizer) -> dict:
        t0 = time.time()
        results = jax.device_get(run_all(actor_params, normalizer))
        per_map = {}
        for name, r in zip(map_names, results):
            complete = r["lap"] & ~r["collision"]
            lap_times = r["length"][complete] * cp.POLICY_DT
            per_map[name] = {
                "episodes": int(len(complete)),
                "completion_rate": float(complete.mean()),
                "collision_rate": float(r["collision"].mean()),
                "timeout_rate": float(r["timeout"].mean()),
                "lap_time_mean": float(lap_times.mean()) if len(lap_times) else None,
                "lap_time_min": float(lap_times.min()) if len(lap_times) else None,
                "mean_speed": float(r["speed"].mean()),
                "mean_progress_laps": float(r["progress"].mean()),
                "per_episode": {
                    "complete": complete.tolist(),
                    "collision": r["collision"].tolist(),
                    "length_steps": r["length"].tolist(),
                    "progress_laps": r["progress"].tolist(),
                },
            }
        rates = [v["completion_rate"] for v in per_map.values()]
        return {
            "setting": "randomized" if randomize else "nominal",
            "maps": list(map_names),
            "episodes_per_map": episodes_per_map,
            "max_steps": max_steps,
            "completion_rate": float(np.mean(rates)),
            "collision_rate": float(np.mean([v["collision_rate"] for v in per_map.values()])),
            "mean_progress_laps": float(np.mean([v["mean_progress_laps"] for v in per_map.values()])),
            "per_map": per_map,
            "eval_seconds": time.time() - t0,
        }

    return evaluate


def main(argv=None):
    from .train_ppo import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--randomize", action="store_true", help="vehicle params, delay, offset and noise on")
    parser.add_argument("--maps", nargs="*", default=list(M.HELDOUT_MAPS))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_EVAL_MAX_STEPS)
    parser.add_argument("--map-dir", default=str(M.DEFAULT_MAP_DIR))
    parser.add_argument("--out", default=None, help="JSON report path (default: next to the checkpoint)")
    args = parser.parse_args(argv)

    M.use_map_dir(args.map_dir)
    ckpt = load_checkpoint(args.checkpoint)
    evaluate = make_evaluator(args.maps, args.episodes, args.max_steps, args.randomize,
                              hidden_dim=ckpt["config"]["hidden_dim"])
    report = evaluate(ckpt["actor_params"], ckpt["normalizer"])
    report["checkpoint"] = str(args.checkpoint)
    report["checkpoint_sha256"] = ckpt["sha256"]
    out = pathlib.Path(args.out) if args.out else pathlib.Path(args.checkpoint).with_suffix(
        f".eval_{report['setting']}.json"
    )
    out.write_text(json.dumps(report, indent=2))
    summary = {m: round(v["completion_rate"], 2) for m, v in report["per_map"].items()}
    print(f"{report['setting']} completion {report['completion_rate']:.2f} {summary} -> {out}")


if __name__ == "__main__":
    main()
