#!/bin/bash
# Phase-2 (1v1) dress rehearsal on one H100, from a phase-1c GRU checkpoint:
# validation tests (CPU JAX) -> race smoke run (256 envs, 10 min) -> 2048-env
# race throughput probe -> race evaluation of the smoke checkpoint ->
# slurm/phase2_budget.json. Proves tomorrow's slurm/phase2.sbatch path.
#
#   sbatch --export=ALL,INIT_FROM=<1c gru ckpt>[,VENV=..,MAP_DIR=..,DETACH=1] slurm/phase2_prep.sbatch
#
#SBATCH --job-name=rl-2-prep
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p slurm/logs
: "${INIT_FROM:?set INIT_FROM to a phase-1c cnn_gru checkpoint}"
export PATH="$HOME/.local/bin:$PATH" XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
export UV_PROJECT_ENVIRONMENT="${VENV:-$PWD/.venv}"
export F1TENTH_GYM_JAX_MAP_DIR="${MAP_DIR:-$PWD/rl/maps}"
ARGS=(--arch cnn_gru --frame-stack 1 --race true --init-from "$INIT_FROM" --critic-warmup 20
      --vf-coef "${VF_COEF:-0.1}" --ent-coef 0.005 --max-delay 4 --wandb-mode offline)
[ "${DETACH:-1}" = 1 ] && ARGS+=(--gru-critic-detach true)
echo "host $(hostname) job ${SLURM_JOB_ID:-local} commit $(git rev-parse --short HEAD) init_from $INIT_FROM detach ${DETACH:-1}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo "=== 1. validation tests (CPU JAX)"
JAX_PLATFORMS=cpu uv run --frozen pytest -q rl/tests 2>&1 | tail -6
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "TESTS FAILED"; exit 1; }
echo "TESTS PASSED"

echo "=== 2. race smoke (256 envs, 10 min)"
rm -rf runs/smoke_2
uv run --frozen python -m rl.train_ppo "${ARGS[@]}" --num-envs 256 --time-limit 600 --eval-every 50 --checkpoint-every 25 \
  --runs-root runs/smoke_2 2>&1 | grep -vE "sol_gpu_cost_model|^wandb"
python3 - <<'PY'
import glob, json
rows = [json.loads(l) for f in glob.glob("runs/smoke_2/*/metrics.jsonl") for l in open(f)]
pick = lambda r: {k: round(r[k], 4) for k in ("update", "episode_return", "collision_rate", "approx_kl",
                  "race/win_rate", "race/car_contact_rate", "race/overtakes", "race/opponent_episodes")}
print("first:", pick(rows[0])); print("last: ", pick(rows[-1]))
nan = any(r["nan_count"] > 0 for r in rows)
print("SMOKE", "FAILED (NaN)" if nan else "OK", f"({len(rows)} updates)")
PY

echo "=== 3. race throughput probe (2048 envs)"
rm -rf runs/probe_2
uv run --frozen python -m rl.train_ppo "${ARGS[@]}" --num-envs 2048 --time-limit 240 --eval-every 100000 \
  --checkpoint-every 100000 --runs-root runs/probe_2 2>&1 | grep -E "^update|setup|initialized"

echo "=== 4. race evaluation of the smoke checkpoint (2 maps, 4 episodes)"
CK=$(ls runs/smoke_2/*/latest.safetensors)
uv run --frozen python -m rl.race_eval --checkpoint "$CK" --maps Duct1077 Spielberg --episodes 4 --max-steps 3000 \
  --out runs/smoke_2/race_eval.json 2>&1 | grep -vE "sol_gpu_cost_model"

python3 - <<'PY'
import glob, json, statistics
rows = [json.loads(l) for f in glob.glob("runs/probe_2/*/metrics.jsonl") for l in open(f)]
sps = statistics.median(r["sps"] for r in rows[1:]) if len(rows) > 1 else rows[0]["sps"]
steps = 2048 * 256
budget = int(sps * (41400 - 600) * 0.9) // steps * steps
json.dump({"race_sps_2048": round(sps), "total_timesteps_12h": budget, "updates": budget // steps},
          open("slurm/phase2_budget.json", "w"), indent=2)
print(f"race sps {sps:.0f} -> 12 h budget {budget:.3g} env-steps")
PY
echo "PHASE2 PREP DONE $(date -Is)"
