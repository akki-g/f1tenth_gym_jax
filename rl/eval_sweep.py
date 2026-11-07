"""Evaluate several checkpoints on held-out and never-used maps.

The held-out maps (``maps.HELDOUT_MAPS``) are unseen by training but pick
``best.safetensors``. The *unseen* set here is used by neither: 12 more
racetracks from the F1TENTH set and 6 narrow tracks from fresh generator
seeds. Each setting (nominal, randomized) compiles one evaluator over all maps
and runs every checkpoint through it with the same start points and noise.

    python -m rl.eval_sweep prepare                     # login node: write/download the unseen maps
    python -m rl.eval_sweep run runs/phase1/<run> --last 4 [--settings nominal randomized]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time

import numpy as np

from . import maps as M

UNSEEN_RACETRACKS = (
    "IMS", "Melbourne", "MexicoCity", "Nuerburgring", "Sakhir", "SaoPaulo",
    "Sepang", "Shanghai", "Sochi", "Spa", "YasMarina", "Zandvoort",
)
UNSEEN_GEN_SEEDS = (3012, 3014, 3020, 3023, 3034, 3053)  # first 6 valid seeds from 3000


def unseen_gen_name(seed: int) -> str:
    return f"EvalDuct{seed}"  # not "Duct": keeps them apart from the train/held-out tracks (and out of git)


UNSEEN_MAPS = UNSEEN_RACETRACKS + tuple(unseen_gen_name(s) for s in UNSEEN_GEN_SEEDS)
assert not set(UNSEEN_MAPS) & set(M.TRAIN_MAPS + M.HELDOUT_MAPS)
GROUPS = {"heldout": M.HELDOUT_MAPS, "unseen": UNSEEN_MAPS}
# lab suite: lab-size tracks only (held-out: picks checkpoints; unseen: used by neither)
SUITES = {"phase1c": GROUPS,
          "lab": {"heldout": M.LAB_FT_HELDOUT_MAPS, "unseen": M.LAB_EVAL_UNSEEN_MAPS, "lastrace": M.LAST_RACE_MAPS}}


def prepare_unseen(map_dir=M.DEFAULT_MAP_DIR) -> dict:
    """Write the fresh generated tracks, download the extra racetracks, record hashes."""
    from f1tenth_gym_jax.envs.track.utils import find_track_dir

    map_dir = M.use_map_dir(map_dir)
    manifest = {"maps": {}}
    for seed in UNSEEN_GEN_SEEDS:
        name = unseen_gen_name(seed)
        if not (map_dir / name / f"{name}.png").exists():
            g = M.generate_duct_track(seed)
            if g is None:
                raise RuntimeError(f"seed {seed} no longer yields a valid track")
            M.write_map(map_dir, name, g["free"], g["resolution"], g["origin"], g["centerline"])
    for name in UNSEEN_MAPS:
        track_dir = find_track_dir(name)  # downloads racetracks if missing
        manifest["maps"][name] = {
            p.name: M._sha256(p) for p in sorted(track_dir.iterdir())
            if p.suffix in (".png", ".pgm", ".yaml", ".csv") and p.name.startswith(track_dir.name)
        }
    (map_dir / "unseen_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _update_of(path: pathlib.Path) -> int:
    m = re.search(r"ckpt_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def run(checkpoints, settings, episodes, max_steps, out_dir: pathlib.Path, suite: str = "phase1c",
        lab_physics: bool = False) -> dict:
    """``lab_physics``: evaluate under the lab fine-tune's physics (rollover,
    carpet friction, 0-240 ms delay) whatever the checkpoints trained with."""
    from . import car_params as cp
    from .evaluate import friction_from, make_evaluator, wrapper_kwargs_from
    from .train_ppo import load_checkpoint

    groups = SUITES[suite]

    loaded = [(p, load_checkpoint(p)) for p in checkpoints]
    from .networks import model_cfg_from

    cfgs = {json.dumps(model_cfg_from(ck["config"]), sort_keys=True) for _, ck in loaded}
    if len(cfgs) != 1:
        raise ValueError(f"checkpoints disagree on the model: {cfgs}")
    model_cfg = model_cfg_from(loaded[0][1]["config"])
    map_names = tuple(n for names in groups.values() for n in names)
    cfg = loaded[-1][1]["config"]
    wrapper_kwargs, friction = wrapper_kwargs_from(cfg), friction_from(cfg)
    if lab_physics:
        wrapper_kwargs.update(rollover=True, steer_asym=True, max_delay=cp.LAB_MAX_DELAY_STEPS)
        friction = cp.LAB_FRICTION
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for setting in settings:
        t0 = time.time()
        evaluate = make_evaluator(map_names, episodes, max_steps, randomize=(setting == "randomized"),
                                  model_cfg=model_cfg, wrapper_kwargs=wrapper_kwargs, friction=friction)
        print(f"[{setting}] evaluator built in {time.time() - t0:.0f}s", flush=True)
        for path, ck in loaded:
            rep = evaluate(ck["actor_params"], ck["normalizer"])
            rep.update(checkpoint=str(path), checkpoint_sha256=ck["sha256"], update=ck["update"])
            row = {"checkpoint": path.name, "update": ck["update"], "setting": setting,
                   "eval_seconds": round(rep["eval_seconds"], 1)}
            for group, names in groups.items():
                per = [rep["per_map"][n] for n in names]
                row[f"{group}_completion"] = float(np.mean([v["completion_rate"] for v in per]))
                row[f"{group}_collision"] = float(np.mean([v["collision_rate"] for v in per]))
                row[f"{group}_rollover"] = float(np.mean([v.get("rollover_rate", 0.0) for v in per]))
                row[f"{group}_worst"] = min((v["completion_rate"], n) for v, n in zip(per, names))
            rep["groups"] = {g: list(n) for g, n in groups.items()}
            rep.update(suite=suite, lab_physics=lab_physics)
            (out_dir / f"{setting}_{path.stem}.json").write_text(json.dumps(rep, indent=2))
            summary.append(row)
            print(f"[{setting}] {path.name} (update {ck['update']}): "
                  f"held-out {row['heldout_completion']:.2f} (worst {row['heldout_worst'][1]} {row['heldout_worst'][0]:.1f}) | "
                  f"unseen {row['unseen_completion']:.2f} (worst {row['unseen_worst'][1]} {row['unseen_worst'][0]:.1f}) "
                  f"| {row['eval_seconds']}s", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"rows": summary}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--map-dir", default=str(M.DEFAULT_MAP_DIR))
    r = sub.add_parser("run")
    r.add_argument("run_dir", type=pathlib.Path, help="training run directory (ckpt_*.safetensors)")
    r.add_argument("--last", type=int, default=4, help="the N most recent periodic checkpoints")
    r.add_argument("--checkpoints", nargs="*", default=None, help="explicit checkpoint paths instead of --last")
    r.add_argument("--settings", nargs="+", default=["nominal", "randomized"], choices=["nominal", "randomized"])
    r.add_argument("--episodes", type=int, default=10)
    r.add_argument("--max-steps", type=int, default=6000)
    r.add_argument("--map-dir", default=str(M.DEFAULT_MAP_DIR))
    r.add_argument("--out-dir", type=pathlib.Path, default=None)
    r.add_argument("--suite", choices=sorted(SUITES), default="phase1c")
    r.add_argument("--lab-physics", action="store_true",
                   help="rollover, carpet friction, 0-240 ms delay, measured left/right steering")
    args = parser.parse_args(argv)

    if args.cmd == "prepare":
        man = prepare_unseen(args.map_dir)
        print(f"prepared {len(man['maps'])} unseen maps in {args.map_dir}")
        return
    M.use_map_dir(args.map_dir)
    if args.checkpoints:
        ckpts = [pathlib.Path(c) for c in args.checkpoints]
    else:
        ckpts = sorted(args.run_dir.glob("ckpt_*.safetensors"), key=_update_of)[-args.last:]
    if not ckpts:
        raise SystemExit(f"no checkpoints found in {args.run_dir}")
    out = args.out_dir or pathlib.Path("runs/eval_sweep") / args.run_dir.name
    print("checkpoints:", [c.name for c in ckpts], "->", out, flush=True)
    run(ckpts, args.settings, args.episodes, args.max_steps, out, suite=args.suite, lab_physics=args.lab_physics)


if __name__ == "__main__":
    main()
