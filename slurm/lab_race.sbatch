#!/bin/bash
# Lab race fine-tune (phase 2 on lab tracks): two cars, an opponent drawn per episode (none 20%, a past
# self 35%, follow-the-gap 15%, racing-line pursuit 30%, half of those slow cars at 0.7-1.3 m/s), race
# reward, privileged critic; lab tracks, rollover, carpet friction, 0-240 ms delay, 5 laps per episode.
# From a phase-1c cnn_gru checkpoint, on one H100, until the time limit (clean stop with a checkpoint).
#
#   sbatch --export=ALL,INIT_FROM=runs/phase1c/<gru run>/ckpt_02050.safetensors slurm/lab_race.sbatch
#   options (env): LR (1e-4), VF_COEF (0.1), TRAIN_SECONDS (36000), SEED (0), VENV, MAP_DIR
#
#SBATCH --job-name=rl-lab-race
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=10:30:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm/logs
: "${INIT_FROM:?set INIT_FROM to a phase-1c cnn_gru checkpoint}"
export PATH="$HOME/.local/bin:$PATH" XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
export UV_PROJECT_ENVIRONMENT="${VENV:-$PWD/.venv}"
export F1TENTH_GYM_JAX_MAP_DIR="${MAP_DIR:-$PWD/rl/maps}"
TRAIN_SECONDS="${TRAIN_SECONDS:-36000}"
echo "host $(hostname) job ${SLURM_JOB_ID:-local} commit $(git rev-parse --short HEAD) init_from $INIT_FROM"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

uv run --frozen python -m rl.maps prepare --lab --map-dir "$F1TENTH_GYM_JAX_MAP_DIR" | tail -1
read -r -a TRAIN <<< "$(uv run --frozen python -c 'from rl import maps as M; print(" ".join(M.LAB_FT_TRAIN_MAPS))')"
read -r -a HELDOUT <<< "$(uv run --frozen python -c 'from rl import maps as M; print(" ".join(M.LAB_FT_HELDOUT_MAPS))')"
echo "=== race train on ${#TRAIN[@]} maps (until $((TRAIN_SECONDS - 300)) s)"
uv run --frozen python -m rl.train_ppo --arch cnn_gru --frame-stack 1 --race true --init-from "$INIT_FROM" \
  --critic-warmup 20 --lr "${LR:-1e-4}" --anneal-lr false --vf-coef "${VF_COEF:-0.1}" --ent-coef 0.005 \
  --seed "${SEED:-0}" --num-envs 2048 --max-delay 6 --rollover true --steer-asym true --friction-lo 1.1 --friction-hi 1.5 \
  --max-laps 5 --train-maps "${TRAIN[@]}" --heldout-maps "${HELDOUT[@]}" --wandb-mode offline \
  --total-timesteps 20000000000 --time-limit "$((TRAIN_SECONDS - 300))" --runs-root runs/lab_race
echo "done $(date -Is)"
