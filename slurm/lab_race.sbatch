#!/bin/bash
# Lab race fine-tune on the lab maps: two cars, with RACE_MODE
#   opponents  an opponent per episode (none 20%, a past self 35% (GRU only; the MLP gets racing-line
#              pursuit instead), follow-the-gap 15%, racing-line pursuit 30%, half of those slow cars)
#   obstacles  a parked car where the corridor leaves room to pass (80% of episodes, else solo)
#   mixed      odd-numbered maps obstacles, the others opponents
# plus rollover, carpet friction, 0-240 ms delay, the car's steering, random direction per episode,
# 5.5-6.5 m/s^2 acceleration, 5 laps per episode. From a phase-1c checkpoint, on one GPU (H100 or V100), until
# the time limit (clean stop with a checkpoint).
#
#   sbatch --export=ALL,INIT_FROM=runs/phase1c/<run>/ckpt_02050.safetensors,RACE_MODE=opponents slurm/lab_race.sbatch
#   options (env): ARCH (cnn_gru | mlp), FRAME_STACK (1 / 3), RACE_MODE, LR (1e-4), VF_COEF (0.1),
#                  TRAIN_SECONDS (37800), SEED (0), VENV, MAP_DIR
#
#SBATCH --job-name=rl-lab-race
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
# JAX runs from its CUDA 12 build (.venv-cu12, as in train_phase1.sbatch); a run uses ~8 GB.
case "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" in
  *H100*) DEFAULT_VENV="$PWD/.venv" ;;
  *)      DEFAULT_VENV="$PWD/.venv-cu12" ;;
esac
export UV_PROJECT_ENVIRONMENT="${VENV:-$DEFAULT_VENV}"
export F1TENTH_GYM_JAX_MAP_DIR="${MAP_DIR:-$PWD/rl/maps}"
TRAIN_SECONDS="${TRAIN_SECONDS:-37800}"
ARCH="${ARCH:-cnn_gru}"
FRAME_STACK="${FRAME_STACK:-$([ "$ARCH" = cnn_gru ] && echo 1 || echo 3)}"
RACE_MODE="${RACE_MODE:-opponents}"
echo "host $(hostname) job ${SLURM_JOB_ID:-local} commit $(git rev-parse --short HEAD) arch $ARCH mode $RACE_MODE init_from $INIT_FROM"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "env $UV_PROJECT_ENVIRONMENT"

uv run --frozen python -m rl.maps prepare --lab --map-dir "$F1TENTH_GYM_JAX_MAP_DIR" | tail -1
read -r -a TRAIN <<< "$(uv run --frozen python -c 'from rl import maps as M; print(" ".join(M.LAB_FT_TRAIN_MAPS))')"
read -r -a HELDOUT <<< "$(uv run --frozen python -c 'from rl import maps as M; print(" ".join(M.LAB_FT_HELDOUT_MAPS))')"
echo "=== race train ($RACE_MODE) on ${#TRAIN[@]} maps (until $((TRAIN_SECONDS - 300)) s)"
uv run --frozen python -m rl.train_ppo --arch "$ARCH" --frame-stack "$FRAME_STACK" --race true --race-mode "$RACE_MODE" \
  --init-from "$INIT_FROM" --critic-warmup 20 --lr "${LR:-1e-4}" --anneal-lr false --vf-coef "${VF_COEF:-0.1}" \
  --ent-coef 0.005 --seed "${SEED:-0}" --num-envs 2048 --max-delay 6 --rollover true --steer-asym true \
  --friction-lo 1.1 --friction-hi 1.5 --max-laps 5 --random-direction true --accel-lo 5.5 --accel-hi 6.5 \
  --train-maps "${TRAIN[@]}" --heldout-maps "${HELDOUT[@]}" --wandb-mode offline \
  --total-timesteps 20000000000 --time-limit "$((TRAIN_SECONDS - 300))" --runs-root runs/lab_race
echo "done $(date -Is)"
