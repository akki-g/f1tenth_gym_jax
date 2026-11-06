"""Phase-2 (1v1) evaluation: win rate, passes, contact and solo lap time.

Recurrent-policy spec, "Phase 2" evaluation: held-out maps, side-by-side and
staggered starts, against each opponent type at speed factor 0.9, plus a
held-out opponent (follow-the-gap at an unseen speed factor, 1.1), and solo
episodes for the lap-time check. The ego is deterministic (tanh(mean));
episodes are 1 lap; each env's first episode is scored.

    python -m rl.race_eval --checkpoint runs/<race run>/best.safetensors [--snapshot <1c or race ckpt>]
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
from .networks import CnnGru, deterministic_action, model_cfg_from
from .race import OPP_FTG, OPP_NAMES, OPP_PURSUIT, OPP_SNAPSHOT, OPP_SOLO, PRIV_DIM, RaceWrapper

DEFAULT_SETTINGS = (  # (name, opponent, speed factor)
    ("follow_gap_0.9", OPP_FTG, 0.9),
    ("pursuit_0.9", OPP_PURSUIT, 0.9),
    ("snapshot", OPP_SNAPSHOT, 1.0),
    ("heldout_follow_gap_1.1", OPP_FTG, 1.1),
    ("solo", OPP_SOLO, 1.0),
)


def make_race_evaluator(map_names=M.HELDOUT_MAPS, episodes: int = 10, max_steps: int = 6000, model_cfg=None,
                        settings=DEFAULT_SETTINGS, seed: int = 4321):
    from f1tenth_gym_jax.envs import F110Env

    from .domain_rand import make_param

    mc = model_cfg or model_cfg_from({"arch": "cnn_gru", "num_beams": cp.NUM_BEAMS, "frame_stack": 1})
    model = CnnGru(mc["num_beams"], mc["gru_dim"], mc["proj_dim"], log_std_init=mc["log_std_init"], priv_dim=PRIV_DIM)
    cases = []
    for m in map_names:
        for name, opp, scale in settings:
            for side in ((True, False) if opp != OPP_SOLO else (False,)):
                env = F110Env(2, make_param(m, randomize=False, max_steps=max_steps, scan_other_agents=True,
                                            num_beams=mc["num_beams"]))
                w = RaceWrapper(env, randomize=False, frame_stack=1, max_laps=1, gru_dim=mc["gru_dim"],
                                snapshot_model=model, fixed_opponent=opp, fixed_speed_scale=scale,
                                fixed_side_by_side=side)
                cases.append((m, name, "side_by_side" if side else ("staggered" if opp != OPP_SOLO else "solo"), w))

    def run_case(w, params, normalizer, pool, key):
        n = episodes
        obs, rs = jax.vmap(w.reset)(jax.random.split(key, n))
        rs = jax.tree.map(lambda x: x.astype(x.dtype), rs)
        init = dict(t=0, obs=obs, rs=rs, h=jnp.zeros((n, mc["gru_dim"])), first=jnp.ones(n, bool), key=key,
                    finished=jnp.zeros(n, bool), won=jnp.zeros(n, bool), lap=jnp.zeros(n, bool),
                    collision=jnp.zeros(n, bool), car_contact=jnp.zeros(n, bool), overtakes=jnp.zeros(n, jnp.int32),
                    length=jnp.zeros(n))

        def cond(c):
            return (c["t"] < max_steps + 1) & ~jnp.all(c["finished"])

        def body(c):
            key, k = jax.random.split(c["key"])
            h_in = jnp.where(c["first"][:, None], 0.0, c["h"])
            h, mean, _, _ = model.apply(params, h_in, normalizer(c["obs"]))
            obs, rs, _, done, info = jax.vmap(w.step, in_axes=(0, 0, 0, None))(
                jax.random.split(k, n), c["rs"], deterministic_action(mean), pool)
            first = done & ~c["finished"]
            rec = lambda old, new: jnp.where(first, new, old)  # noqa: E731
            return dict(t=c["t"] + 1, obs=obs, rs=rs, h=h, first=done, key=key, finished=c["finished"] | done,
                        won=rec(c["won"], info["won"]), lap=rec(c["lap"], info["lap_complete"]),
                        collision=rec(c["collision"], info["collision"]),
                        car_contact=rec(c["car_contact"], info["car_contact"]),
                        overtakes=rec(c["overtakes"], info["overtakes"]),
                        length=rec(c["length"], info["episode_length"].astype(jnp.float32)))

        out = jax.lax.while_loop(cond, body, init)
        return {k: out[k] for k in ("won", "lap", "collision", "car_contact", "overtakes", "length")}

    run = jax.jit(run_case, static_argnums=0)

    def evaluate(params, normalizer, pool) -> dict:
        t0 = time.time()
        rows = []
        for i, (m, name, start, w) in enumerate(cases):
            r = jax.device_get(run(w, params, normalizer, pool, jax.random.PRNGKey(seed + i)))
            complete = r["lap"] & ~r["collision"]
            lap_s = r["length"][complete] * cp.POLICY_DT
            rows.append({"map": m, "opponent": name, "start": start, "episodes": episodes,
                         "win_rate": float(r["won"].mean()), "contact_rate": float(r["car_contact"].mean()),
                         "overtakes": float(r["overtakes"].mean()), "completion": float(complete.mean()),
                         "lap_time": float(lap_s.mean()) if len(lap_s) else None})
        summary = {}
        for name, _, _ in settings:
            sel = [x for x in rows if x["opponent"] == name]
            summary[name] = {k: float(np.mean([x[k] for x in sel if x[k] is not None])) if any(x[k] is not None for x in sel) else None
                             for k in ("win_rate", "contact_rate", "overtakes", "completion", "lap_time")}
        return {"maps": list(map_names), "episodes": episodes, "summary": summary, "cases": rows,
                "eval_seconds": time.time() - t0}

    return evaluate


def snapshot_pool(params, normalizer):
    """A one-slot opponent pool holding the given (race-format) policy."""
    one = lambda x: jnp.asarray(x)[None]  # noqa: E731
    return {"params": jax.tree.map(one, params), "normalizer": jax.tree.map(one, normalizer)}


def main(argv=None):
    from .train_ppo import load_checkpoint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="race (phase-2) checkpoint")
    ap.add_argument("--snapshot", default=None, help="opponent for the snapshot setting (default: the checkpoint itself)")
    ap.add_argument("--maps", nargs="*", default=list(M.HELDOUT_MAPS))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--map-dir", default=str(M.DEFAULT_MAP_DIR))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    M.use_map_dir(a.map_dir)
    ck = load_checkpoint(a.checkpoint)
    opp = load_checkpoint(a.snapshot) if a.snapshot else ck
    evaluate = make_race_evaluator(a.maps, a.episodes, a.max_steps, model_cfg_from(ck["config"]))
    rep = evaluate(ck["actor_params"], ck["normalizer"], snapshot_pool(opp["actor_params"], opp["normalizer"]))
    rep.update(checkpoint=str(a.checkpoint), checkpoint_sha256=ck["sha256"], snapshot=str(a.snapshot or a.checkpoint))
    out = pathlib.Path(a.out) if a.out else pathlib.Path(a.checkpoint).with_suffix(".race_eval.json")
    out.write_text(json.dumps(rep, indent=2))
    for name, v in rep["summary"].items():
        print(f"{name:24s} " + "  ".join(f"{k} {'-' if x is None else round(x, 3)}" for k, x in v.items()))
    print("->", out)


if __name__ == "__main__":
    main()
