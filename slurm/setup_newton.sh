#!/bin/bash
# One-time setup on the Newton login node (internet access needed).
#   bash slurm/setup_newton.sh
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# GPU build of JAX. The cuda13 wheels need an NVIDIA driver >= 580; check
# with `nvidia-smi` on a GPU node. With an older driver, install the CUDA 12
# wheels instead:  uv pip install "jax[cuda12]==0.7.2"
uv sync --frozen --extra cuda --extra rl --group dev --group parity

# tracks: write the generated ones, download the racetracks, record hashes
uv run --frozen python -m rl.maps prepare

# the car repo next to this one: the shared obs.py / numpy_policy.py checks compare against it
[ -d ../roboracer_ros2 ] || git clone -b rl-phase1 https://github.com/akki-g/roboracer_ros2.git ../roboracer_ros2

# validation gate (spec section 6): every check must pass before training.
# CPU is fine for this; the parity test clones the classic f1tenth_gym.
JAX_PLATFORMS=cpu uv run --frozen pytest -q rl/tests
echo "setup complete; submit the smoke run:"
echo "  sbatch --time=00:20:00 --export=ALL,MODE=smoke slurm/train_phase1.sbatch"
