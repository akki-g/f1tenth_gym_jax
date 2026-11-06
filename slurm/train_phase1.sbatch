#!/bin/bash
# Phase-1 PPO training on one H100 (spec section 5, "Slurm job").
#
#   Smoke run (10 min, 256 envs):
#     sbatch --time=00:20:00 --export=ALL,MODE=smoke slurm/train_phase1.sbatch
#   12-hour job (full config; set TOTAL_TIMESTEPS from the smoke run's sps):
#     sbatch --export=ALL,MODE=full,TOTAL_TIMESTEPS=1.18e9 slurm/train_phase1.sbatch
#   (1.18e9 = 31,736 steps/s measured at 2048 envs on a Newton H100 PCIe x 41400 s x 0.9)
#
# Restarting (requeue, or the same sbatch line again) resumes from the run
# directory's latest checkpoint: the directory is named <date>_<config hash>
# and train_ppo.py picks up an existing one with the same hash.
#
# One-time setup on the login node first: bash slurm/setup_newton.sh, then
# the validation gate (slurm/gate_tests.sbatch); chain the smoke run on it.
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
# Submitted with --constraint="h100|gpu32" a job may land on a V100 (32 GB).
# CUDA 13 dropped Volta, so V100s use the CUDA 12 build of the same JAX
# (.venv-cu12, built once on the login node). Same env-step budget either way;
# a V100 just needs a longer wall clock (TRAIN_TIME_LIMIT_OTHER, and --time).
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
case "$GPU_NAME" in
  *H100*) TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT_H100:-41400}" ;;
  *)      export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cu12"
          TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT_OTHER:-41400}" ;;
esac
echo "gpu '$GPU_NAME' -> env ${UV_PROJECT_ENVIRONMENT:-.venv}, training time limit ${TRAIN_TIME_LIMIT}s"
uv run --frozen python -c "import jax; print('jax', jax.__version__, jax.devices()); assert jax.default_backend() == 'gpu'"

# the tracks must already be on disk (compute nodes may have no internet)
uv run --frozen python - <<'PY'
import hashlib, json, pathlib, sys
from f1tenth_gym_jax.envs.track.utils import get_map_search_dirs  # where the env looks; never downloads
m = pathlib.Path("rl/maps/manifest.json")
if not m.exists():
    sys.exit("rl/maps/manifest.json missing: run `uv run python -m rl.maps prepare` on the login node")
maps = json.loads(m.read_text())["maps"]
dirs = get_map_search_dirs()
bad = []
for name, entry in maps.items():  # the manifest is in git, the racetracks are not: check the files
    found = next((d / name for d in dirs if (d / name).is_dir()), None)
    if found is None:
        bad.append(f"{name}: not found in {[str(d) for d in dirs]}")
        continue
    for fname, digest in entry["files"].items():
        p = found / fname
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            bad.append(f"{name}/{fname}: missing or hash differs from the manifest")
if bad:
    sys.exit("track check failed (run `uv run python -m rl.maps prepare` on the login node):\n  " + "\n  ".join(bad))
print("maps:", len(maps), "all files match the manifest")
PY

# TOTAL_TIMESTEPS=auto: the shared phase-1c budget written by
# slurm/phase1c_prep.sbatch from its 2048-env throughput probes
if [ "${TOTAL_TIMESTEPS:-}" = auto ]; then
  TOTAL_TIMESTEPS=$(python3 -c "import json; print(json.load(open('slurm/phase1c_budget.json'))['total_timesteps'])")
  echo "budget from slurm/phase1c_budget.json: $TOTAL_TIMESTEPS env-steps"
fi

case "$MODE" in
  smoke)
    # 10 minutes of training (+ compile, final eval and checkpoint) with 256 envs
    RUNS_ROOT="${RUNS_ROOT:-runs/smoke}"
    ARGS=(--num-envs 256 --time-limit 600 --eval-every 50 --checkpoint-every 25
          --runs-root "$RUNS_ROOT")
    ;;
  full)
    # spec hyperparameters; stop cleanly ~20 min before the wall clock
    RUNS_ROOT="${RUNS_ROOT:-runs/phase1}"
    ARGS=(--num-envs 2048 --total-timesteps "${TOTAL_TIMESTEPS:-2e8}" --time-limit "$TRAIN_TIME_LIMIT"
          --runs-root "$RUNS_ROOT")
    ;;
  *) echo "unknown MODE=$MODE (smoke|full)"; exit 2 ;;
esac

# no srun: under srun the smoke job hung for 7 min after training finished
# (srun keeps the step open while a leftover child holds its output stream);
# launched directly, the same run exits cleanly. One task, so srun adds nothing.
uv run --frozen python -m rl.train_ppo "${ARGS[@]}" --wandb-mode offline "$@"

if [ "$MODE" = smoke ]; then
  RUN_DIR=$(ls -td "$RUNS_ROOT"/*/ | head -1)
  uv run --frozen python -m rl.check_smoke "$RUN_DIR" || true
fi
echo "done $(date -Is)"
