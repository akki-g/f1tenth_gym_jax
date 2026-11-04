#!/bin/bash
# Phase-1 PPO training on one H100 (spec section 5, "Slurm job").
#
#   Smoke run (10 min, 256 envs):
#     sbatch --time=00:20:00 --export=ALL,MODE=smoke slurm/train_phase1.sbatch
#   12-hour job (full config; set TOTAL_TIMESTEPS from the smoke run's sps):
#     sbatch --export=ALL,MODE=full,TOTAL_TIMESTEPS=2e8 slurm/train_phase1.sbatch
#
# Restarting (requeue, or the same sbatch line again) resumes from the run
# directory's latest checkpoint: the directory is named <date>_<config hash>
# and train_ppo.py picks up an existing one with the same hash.
#
# One-time setup on the login node first: bash slurm/setup_newton.sh
#
#SBATCH --job-name=rl-phase1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100          # check the cluster's GPU feature name (sinfo -o "%N %G %f")
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x-%j.out
##SBATCH --partition=<gpu partition>   # uncomment and set if the cluster needs one
##SBATCH --account=<account>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm/logs

MODE="${MODE:-full}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline
export F1TENTH_GYM_JAX_MAP_DIR="$PWD/rl/maps"   # prepared on the login node; no downloads here
# module load cuda   # not needed: the jax[cuda13] wheels bundle CUDA; uncomment if the site requires it

echo "host $(hostname)  job ${SLURM_JOB_ID:-local}  mode $MODE  start $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
uv run --frozen python -c "import jax; print('jax', jax.__version__, jax.devices()); assert jax.default_backend() == 'gpu'"

# the tracks must already be on disk (compute nodes may have no internet)
uv run --frozen python - <<'PY'
import json, pathlib, sys
m = pathlib.Path("rl/maps/manifest.json")
if not m.exists():
    sys.exit("rl/maps/manifest.json missing: run `uv run python -m rl.maps prepare` on the login node")
print("maps:", len(json.loads(m.read_text())["maps"]))
PY

case "$MODE" in
  smoke)
    # 10 minutes of training (+ compile, final eval and checkpoint) with 256 envs
    ARGS=(--num-envs 256 --time-limit 600 --eval-every 50 --checkpoint-every 25
          --runs-root runs/smoke)
    ;;
  full)
    # spec hyperparameters; stop cleanly ~20 min before the wall clock
    ARGS=(--num-envs 2048 --total-timesteps "${TOTAL_TIMESTEPS:-2e8}" --time-limit 41400
          --runs-root runs/phase1)
    ;;
  *) echo "unknown MODE=$MODE (smoke|full)"; exit 2 ;;
esac

srun uv run --frozen python -m rl.train_ppo "${ARGS[@]}" --wandb-mode offline "$@"

if [ "$MODE" = smoke ]; then
  RUN_DIR=$(ls -td runs/smoke/*/ | head -1)
  uv run --frozen python -m rl.check_smoke "$RUN_DIR" || true
fi
echo "done $(date -Is)"
