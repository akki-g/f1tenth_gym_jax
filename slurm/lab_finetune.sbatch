#!/bin/bash
# Lab fine-tune: continue a phase-1c policy on lab-size tracks (10-20 m loops,
# 0.9-2.0 m wide, uneven duct walls) with rollover, carpet friction (1.1-1.5)
# and 0-240 ms action delay, on one H100. Probes the env's speed first, sizes
# the env-step budget for the rest of the allocation, then trains.
#
#   sbatch --export=ALL,INIT_FROM=runs/phase1c/<run>/ckpt_01025.safetensors slurm/lab_finetune.sbatch
#   options (env): ARCH (cnn_gru | mlp), FRAME_STACK (1 for cnn_gru, 3 for the MLP-3), LR (1e-4),
#                  TRAIN_SECONDS (21600), SEED (0), VENV, MAP_DIR
#
#SBATCH --job-name=rl-lab-ft
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:45:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm/logs
: "${INIT_FROM:?set INIT_FROM to a phase-1c checkpoint}"
export PATH="$HOME/.local/bin:$PATH" XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
export UV_PROJECT_ENVIRONMENT="${VENV:-$PWD/.venv}"
export F1TENTH_GYM_JAX_MAP_DIR="${MAP_DIR:-$PWD/rl/maps}"
TRAIN_SECONDS="${TRAIN_SECONDS:-21600}"
ARCH="${ARCH:-cnn_gru}"
FRAME_STACK="${FRAME_STACK:-$([ "$ARCH" = cnn_gru ] && echo 1 || echo 3)}"
echo "host $(hostname) job ${SLURM_JOB_ID:-local} commit $(git rev-parse --short HEAD) arch $ARCH init_from $INIT_FROM"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "=== lab tracks"
uv run --frozen python -m rl.maps prepare --lab --map-dir "$F1TENTH_GYM_JAX_MAP_DIR" | tail -1
read -r -a TRAIN <<< "$(uv run --frozen python -c 'from rl import maps as M; print(" ".join(M.LAB_FT_TRAIN_MAPS))')"
read -r -a HELDOUT <<< "$(uv run --frozen python -c 'from rl import maps as M; print(" ".join(M.LAB_FT_HELDOUT_MAPS))')"
ARGS=(--arch "$ARCH" --frame-stack "$FRAME_STACK" --init-from "$INIT_FROM"
      --lr "${LR:-1e-4}" --vf-coef 0.1 --ent-coef 0.005 --seed "${SEED:-0}" --num-envs 2048
      --max-delay 6 --rollover true --friction-lo 1.1 --friction-hi 1.5 --max-laps 5
      --train-maps "${TRAIN[@]}" --heldout-maps "${HELDOUT[@]}" --wandb-mode offline)

echo "=== throughput probe (2048 envs, ${#TRAIN[@]} maps)"
PROBE="runs/probe_lab_${SLURM_JOB_ID:-local}"
rm -rf "$PROBE"
uv run --frozen python -m rl.train_ppo "${ARGS[@]}" --time-limit 240 --eval-every 100000 --checkpoint-every 100000 \
  --runs-root "$PROBE" 2>&1 | grep -E "^update|setup|initialized|Error|error"
BUDGET=$(python3 - "$TRAIN_SECONDS" "$PROBE" <<'PY'
import glob, json, statistics, sys
rows = [json.loads(l) for f in glob.glob(f"{sys.argv[2]}/*/metrics.jsonl") for l in open(f)]
sps = statistics.median(r["sps"] for r in rows[1:]) if len(rows) > 1 else rows[0]["sps"]
steps = 2048 * 256
budget = int(sps * (float(sys.argv[1]) - 600) * 0.9) // steps * steps  # minus probe and compile time
print(budget)
print(f"lab sps {sps:.0f} -> budget {budget:.3g} env-steps ({budget // steps} updates)", file=sys.stderr)
PY
)
rm -rf "$PROBE"
echo "=== train ($BUDGET env-steps)"
uv run --frozen python -m rl.train_ppo "${ARGS[@]}" --total-timesteps "$BUDGET" --time-limit "$((TRAIN_SECONDS - 600))" \
  --runs-root runs/lab_ft
echo "done $(date -Is)"
