#!/bin/bash
# Lab fine-tune: continue a phase-1c policy on lab-size tracks (10-20 m loops,
# 0.9-2.0 m wide, uneven duct walls) with rollover, carpet friction (1.1-1.5)
# and 0-240 ms action delay, on one GPU (H100 or V100), until the time limit.
#
#   sbatch --export=ALL,INIT_FROM=runs/phase1c/<run>/ckpt_01025.safetensors slurm/lab_finetune.sbatch
#   options (env): ARCH (cnn_gru | mlp), FRAME_STACK (1 for cnn_gru, 3 for the MLP-3), LR (1e-4),
#                  TRAIN_SECONDS (37800), SEED (0), VENV, MAP_DIR
#
#SBATCH --job-name=rl-lab-ft
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|gpu32|gpu16
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=11:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm/logs
: "${INIT_FROM:?set INIT_FROM to a phase-1c checkpoint}"
export PATH="$HOME/.local/bin:$PATH" XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
# H100 or V100 (16/32 GB), whichever frees up first: CUDA 13 dropped Volta, so on a V100 the same
# JAX runs from its CUDA 12 build with cuDNN pinned to 9.8 (.venv-cu12b: jax-cuda12-plugin 0.7.2,
# nvidia-cudnn-cu12==9.8.0.87; cuDNN 9.28, as in .venv-cu12, has no Volta kernels and every
# convolution fails). A run uses ~8 GB.
case "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" in
  *H100*) DEFAULT_VENV="$PWD/.venv" ;;
  *)      DEFAULT_VENV="$PWD/.venv-cu12b" ;;
esac
export UV_PROJECT_ENVIRONMENT="${VENV:-$DEFAULT_VENV}"
export F1TENTH_GYM_JAX_MAP_DIR="${MAP_DIR:-$PWD/rl/maps}"
TRAIN_SECONDS="${TRAIN_SECONDS:-37800}"
ARCH="${ARCH:-cnn_gru}"
FRAME_STACK="${FRAME_STACK:-$([ "$ARCH" = cnn_gru ] && echo 1 || echo 3)}"
echo "host $(hostname) job ${SLURM_JOB_ID:-local} commit $(git rev-parse --short HEAD) arch $ARCH init_from $INIT_FROM"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "env $UV_PROJECT_ENVIRONMENT"

echo "=== lab tracks"
uv run --frozen python -m rl.maps prepare --lab --map-dir "$F1TENTH_GYM_JAX_MAP_DIR" | tail -1
read -r -a TRAIN <<< "$(uv run --frozen python -c 'from rl import maps as M; print(" ".join(M.LAB_FT_TRAIN_MAPS))')"
read -r -a HELDOUT <<< "$(uv run --frozen python -c 'from rl import maps as M; print(" ".join(M.LAB_FT_HELDOUT_MAPS))')"
ARGS=(--arch "$ARCH" --frame-stack "$FRAME_STACK" --init-from "$INIT_FROM"
      --lr "${LR:-1e-4}" --vf-coef 0.1 --ent-coef 0.005 --seed "${SEED:-0}" --num-envs 2048
      --max-delay 6 --rollover true --steer-asym true --friction-lo 1.1 --friction-hi 1.5 --max-laps 5
      --random-direction true --accel-lo 5.5 --accel-hi 6.5
      --train-maps "${TRAIN[@]}" --heldout-maps "${HELDOUT[@]}" --wandb-mode offline)

# No throughput probe: compiling 32 maps takes longer than a short probe, which then sized the budget from
# compile time (17-34 updates). Train until the time limit instead (a clean stop with a checkpoint), at a
# constant learning rate, with a step budget it cannot reach.
echo "=== train (until $((TRAIN_SECONDS - 300)) s)"
uv run --frozen python -m rl.train_ppo "${ARGS[@]}" --anneal-lr false --total-timesteps 20000000000 \
  --time-limit "$((TRAIN_SECONDS - 300))" --runs-root runs/lab_ft
echo "done $(date -Is)"
