"""Check a smoke run against the spec's pass criteria and size the 12 h job.

Pass if returns rise, there are no NaNs, episode lengths grow, the sampled
per-env parameters show variance in the log, and at least one checkpoint and
one held-out evaluation report were written. Prints the measured steps/s and
the total_timesteps that fit in the 12-hour job.

    python -m rl.check_smoke runs/smoke/<run dir>
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

TRAIN_SECONDS_12H = 41400  # --time-limit of the full job
EVAL_OVERHEAD = 0.9  # leave ~10% for held-out evaluations and checkpoints


def check(run_dir: pathlib.Path) -> bool:
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) < 4:
        print(f"FAIL only {len(rows)} updates logged")
        return False
    first, last = rows[: max(2, len(rows) // 5)], rows[-max(2, len(rows) // 5):]

    def avg(rs, key):
        vals = [r[key] for r in rs if r.get("episodes_finished", 0) > 0 and math.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    results = {
        "returns rise": (avg(first, "episode_return"), avg(last, "episode_return")),
        "episode lengths grow": (avg(first, "episode_length"), avg(last, "episode_length")),
    }
    ok = True
    for name, (a, b) in results.items():
        passed = b > a
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'} {name}: {a:.3f} -> {b:.3f}")

    nan = any(not math.isfinite(r["episode_return"]) or r.get("nan_count", 0) > 0 for r in rows)
    ok &= not nan
    print(f"{'PASS' if not nan else 'FAIL'} no NaNs")

    var_keys = ("params/param_mu_std", "params/param_m_std", "params/delay_k_std")
    variances = {k: float(np.nanmean([r[k] for r in rows if r.get("episodes_finished", 0) > 1])) for k in var_keys}
    var_ok = all(v > 0 for v in variances.values())
    ok &= var_ok
    print(f"{'PASS' if var_ok else 'FAIL'} sampled parameters vary: " + ", ".join(f"{k.split('/')[1]}={v:.4f}" for k, v in variances.items()))
    n_eps = sum(1 for _ in (run_dir / "episodes.jsonl").open()) if (run_dir / "episodes.jsonl").exists() else 0
    print(f"     episodes logged with their draws: {n_eps}")

    ckpts = sorted(run_dir.glob("ckpt_*.safetensors"))
    evals = sorted((run_dir / "eval").glob("update_*.json"))
    files_ok = bool(ckpts) and bool(evals)
    ok &= files_ok
    print(f"{'PASS' if files_ok else 'FAIL'} checkpoint and held-out eval written: {len(ckpts)} checkpoints, {len(evals)} eval reports")
    if evals:
        rep = json.loads(evals[-1].read_text())
        print(f"     last eval (update {rep.get('update')}): completion {rep['completion_rate']:.2f}, "
              f"progress {rep['mean_progress_laps']:.3f} laps")

    sps = float(np.median([r["sps"] for r in rows[1:]]))  # skip the compile update
    cfg = json.loads((run_dir / "config.json").read_text())
    print(f"\nsteps/s (median, {cfg['num_envs']} envs): {sps:,.0f}")
    print("For the 2048-env job, steps/s will differ; use its first updates to confirm.")
    budget = sps * TRAIN_SECONDS_12H * EVAL_OVERHEAD
    print(f"total_timesteps that fit in 12 h at this rate: {budget:.3g} (spec default 2e8)")
    print(f"\n{'SMOKE RUN PASSED' if ok else 'SMOKE RUN FAILED'}")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=pathlib.Path)
    args = parser.parse_args(argv)
    sys.exit(0 if check(args.run_dir) else 1)


if __name__ == "__main__":
    main()
