#!/bin/bash
# Phase 2 (1v1) fine-tune from a phase-1c cnn_gru checkpoint, on one H100.
# Probes the race env's speed first (two cars, opponent policies), sizes the
# env-step budget for the rest of the allocation, then trains.
#
#   sbatch --export=ALL,INIT_FROM=runs/phase1c/<gru run>/best.safetensors slurm/phase2.sbatch
#   options (env): VF_COEF (0.1), DETACH (1 = --gru-critic-detach), TRAIN_SECONDS (41400), SEED (0),
#                  VENV, MAP_DIR (reuse another checkout's environment and tracks)
#
#SBATCH --job-name=rl-2-race
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:30:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm/logs
: "${INIT_FROM:?set INIT_FROM to a phase-1c cnn_gru checkpoint}"
export PATH="$HOME/.local/bin:$PATH" XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
# a second checkout (e.g. while phase-1c jobs pin the main one) can reuse the
# main checkout's environment and tracks: VENV=<main>/.venv MAP_DIR=<main>/rl/maps
export UV_PROJECT_ENVIRONMENT="${VENV:-$PWD/.venv}"
export F1TENTH_GYM_JAX_MAP_DIR="${MAP_DIR:-$PWD/rl/maps}"
TRAIN_SECONDS="${TRAIN_SECONDS:-41400}"
ARGS=(--arch cnn_gru --frame-stack 1 --race true --init-from "$INIT_FROM" --critic-warmup 20
      --vf-coef "${VF_COEF:-0.1}" --ent-coef 0.005 --max-delay 4 --seed "${SEED:-0}" --num-envs 2048
      --wandb-mode offline)
[ "${DETACH:-0}" = 1 ] && ARGS+=(--gru-critic-detach true)
echo "host $(hostname) job ${SLURM_JOB_ID:-local} commit $(git rev-parse --short HEAD) init_from $INIT_FROM"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "=== race throughput probe (2048 envs)"
rm -rf runs/probe_2
uv run --frozen python -m rl.train_ppo "${ARGS[@]}" --time-limit 240 --eval-every 100000 --checkpoint-every 100000 \
  --runs-root runs/probe_2 2>&1 | grep -E "^update|setup|initialized"
BUDGET=$(python3 - "$TRAIN_SECONDS" <<'PY'
import glob, json, statistics, sys
rows = [json.loads(l) for f in glob.glob("runs/probe_2/*/metrics.jsonl") for l in open(f)]
sps = statistics.median(r["sps"] for r in rows[1:]) if len(rows) > 1 else rows[0]["sps"]
steps = 2048 * 256
budget = int(sps * (float(sys.argv[1]) - 600) * 0.9) // steps * steps  # minus probe time
print(budget)
print(f"race sps {sps:.0f} -> budget {budget:.3g} env-steps", file=sys.stderr)
PY
)
echo "=== train ($BUDGET env-steps)"
uv run --frozen python -m rl.train_ppo "${ARGS[@]}" --total-timesteps "$BUDGET" --time-limit "$((TRAIN_SECONDS - 600))" \
  --runs-root runs/phase2
echo "done $(date -Is)"
