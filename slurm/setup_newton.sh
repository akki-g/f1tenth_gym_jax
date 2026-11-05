#!/bin/bash
# One-time setup on the Newton login node (internet access needed).
#   bash slurm/setup_newton.sh
#
# Newton's login nodes limit each user to 4 GB of virtual memory and 100
# processes, so this script only installs and downloads; the validation
# tests run as a job (slurm/gate_tests.sbatch) and the smoke run waits on it.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# GPU build of JAX. The cuda13 wheels need an NVIDIA driver >= 580 (Newton's
# H100 nodes: 580.178.04). With an older driver, install the CUDA 12 wheels
# instead:  uv pip install "jax[cuda12]==0.7.2"
# Small thread pools: uv's defaults abort under the login-node ulimits.
export UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLS=2 UV_CONCURRENT_BUILDS=1 \
       RAYON_NUM_THREADS=2 TOKIO_WORKER_THREADS=2
uv sync --frozen --extra cuda --extra rl --group dev --group parity

# tracks: write the generated ones, download the racetracks, record hashes
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 uv run --frozen python -m rl.maps prepare

# the car repo next to this one: the shared obs.py / numpy_policy.py checks compare against it
[ -d ../roboracer_ros2 ] || git clone -b rl-phase1 https://github.com/akki-g/roboracer_ros2.git ../roboracer_ros2

# the classic f1tenth_gym for the parity test (compute nodes may have no internet)
CLASSIC=rl/third_party/f1tenth_gym
if [ ! -d "$CLASSIC/gym/f110_gym" ]; then
  mkdir -p rl/third_party
  git clone -q https://github.com/f1tenth/f1tenth_gym.git "$CLASSIC"
  git -C "$CLASSIC" checkout -q 4fdb9c7e6f
fi

echo "setup complete; run the validation gate, then the smoke run only if every test passes:"
echo '  GATE=$(sbatch --parsable slurm/gate_tests.sbatch)'
echo '  sbatch --dependency=afterok:$GATE --time=00:20:00 --export=ALL,MODE=smoke slurm/train_phase1.sbatch'
