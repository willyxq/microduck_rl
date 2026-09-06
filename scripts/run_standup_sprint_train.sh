#!/usr/bin/env bash
# A: retrain RollerStandUp after servo-index fix
# B: full Sprint training (DuckEMW recipe, 10k iters)
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
export VIRTUAL_ENV="$(pwd)/.venv"
# Avoid picking up an unrelated outer venv.
unset PYTHONHOME || true

NUM_ENVS=2048
LOG=/tmp/standup_sprint_train.log

exec > >(tee -a "$LOG") 2>&1

run_train() {
  local task="$1"
  local run_name="$2"
  shift 2
  echo "============================================================"
  echo "[$(date -Iseconds)] START $task run_name=$run_name $*"
  echo "============================================================"
  uv run train "$task" \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.run-name "$run_name" \
    "$@"
  echo "[$(date -Iseconds)] DONE  $task run_name=$run_name"
}

echo "[$(date -Iseconds)] STANDUP+SPRINT TRAIN QUEUE START"

# A — RollerStandUp (default max_iterations from MicroduckRollerStandUpRlCfg)
run_train Mjlab-RollerStandUp-Flat-MicroDuck roller-standup-refix-2048

# B — Sprint (MicroduckSprintRlCfg.max_iterations=10000)
run_train Mjlab-Sprint-Flat-MicroDuck sprint-full-2048

echo "[$(date -Iseconds)] STANDUP+SPRINT TRAIN QUEUE ALL DONE"
