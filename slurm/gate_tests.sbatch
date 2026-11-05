#!/bin/bash
# Validation gate (spec section 6): every check must pass before training.
# Runs as a job because the login-node ulimits are too small for JAX.
# The tests run on CPU; the GPU is only used to confirm JAX sees it.
#
#   GATE=$(sbatch --parsable slurm/gate_tests.sbatch)
#   sbatch --dependency=afterok:$GATE --time=00:20:00 --export=ALL,MODE=smoke slurm/train_phase1.sbatch
#
#SBATCH --job-name=rl-gate
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm/logs
export PATH="$HOME/.local/bin:$PATH"
export F1TENTH_GYM_JAX_MAP_DIR="$PWD/rl/maps"

echo "host $(hostname)  job ${SLURM_JOB_ID:-local}  start $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
uv run --frozen python -c "import jax; print('jax', jax.__version__, jax.devices()); assert jax.default_backend() == 'gpu'"

set +e
JAX_PLATFORMS=cpu uv run --frozen pytest -q -rA rl/tests 2>&1 | tail -80
rc=${PIPESTATUS[0]}
echo "GATE_EXIT $rc $(date -Is)"
exit $rc
