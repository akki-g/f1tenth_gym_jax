#!/bin/bash
# Phase-1c preparation on ONE H100 allocation (GPUs are scarce, so no queue
# wait between steps): validation gate -> smoke runs (256 envs, 10 min) for
# cnn_gru and mlp-3 -> 2048-env throughput probes for cnn_gru, mlp-3, mlp-8
# -> slurm/phase1c_budget.json (one env-step budget for all three models,
# sized from the slowest probe). Full jobs queued with
#   --dependency=afterok:<this job> --export=ALL,MODE=full,TOTAL_TIMESTEPS=auto
# start as soon as it succeeds. A failed gate test, crash or probe fails the job;
# smoke-check criteria are reported (SMOKE CHECK FAILED) but do not.
#
#SBATCH --job-name=rl-1c-prep
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm/logs
export PATH="$HOME/.local/bin:$PATH" XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
export F1TENTH_GYM_JAX_MAP_DIR="$PWD/rl/maps"
COMMON=(--vf-coef "${VF_COEF:-0.1}" --ent-coef 0.005 --max-delay 4 --wandb-mode offline)
echo "host $(hostname)  job ${SLURM_JOB_ID:-local}  start $(date -Is)  commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "=== 1. validation gate (CPU JAX)"
JAX_PLATFORMS=cpu uv run --frozen pytest -q rl/tests 2>&1 | tail -15
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "GATE FAILED"; exit 1; }
echo "GATE PASSED $(date -Is)"

smoke() {  # name, args...
  local root="runs/smoke_1c/$1"; shift
  rm -rf "$root"
  uv run --frozen python -m rl.train_ppo --num-envs 256 --time-limit 600 --eval-every 50 --checkpoint-every 25 \
    --runs-root "$root" "${COMMON[@]}" "$@" 2>&1 | grep -vE "sol_gpu_cost_model|^wandb"
  # report-only: a failed smoke check is flagged here, not allowed to block the full jobs
  uv run --frozen python -m rl.check_smoke "$(ls -td "$root"/*/ | head -1)" || echo "SMOKE CHECK FAILED: $root"
}
echo "=== 2. smoke runs (256 envs, 10 min)"
smoke cnn_gru --arch cnn_gru --frame-stack 1
smoke mlp3 --arch mlp --frame-stack 3

probe() {  # name, args...
  local root="runs/probe_1c/$1"; shift
  rm -rf "$root"
  uv run --frozen python -m rl.train_ppo --num-envs 2048 --time-limit 240 --eval-every 100000 \
    --checkpoint-every 100000 --runs-root "$root" "${COMMON[@]}" "$@" 2>&1 | grep -E "^update|setup|run dir"
}
echo "=== 3. throughput probes (2048 envs)"
probe cnn_gru --arch cnn_gru --frame-stack 1
probe mlp3 --arch mlp --frame-stack 3
probe mlp8 --arch mlp --frame-stack 8

echo "=== 4. budget"
python3 - <<'PY'
import glob, json, statistics
sps = {}
for name in ("cnn_gru", "mlp3", "mlp8"):
    rows = [json.loads(l) for f in glob.glob(f"runs/probe_1c/{name}/*/metrics.jsonl") for l in open(f)]
    sps[name] = statistics.median(r["sps"] for r in rows[1:])  # skip the compile update
steps = 2048 * 256
budget = int(min(sps.values()) * 41400 * 0.9) // steps * steps
out = {"sps_2048": {k: round(v) for k, v in sps.items()}, "total_timesteps": budget,
       "updates": budget // steps, "rule": "min sps x 41400 s x 0.9, whole updates"}
json.dump(out, open("slurm/phase1c_budget.json", "w"), indent=2)
print(json.dumps(out, indent=2))
PY
echo "PREP DONE $(date -Is)"
