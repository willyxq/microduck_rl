#!/usr/bin/env bash
# Sequential full training: rollers → swizzle → roulade → ball kick (right foot).
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl

NUM_ENVS=2048
LOG=/tmp/multi_gait_train.log
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

# Rollers / swizzle: default max_iterations=50000 in task cfg.
run_train Mjlab-Velocity-Flat-MicroDuck-Rollers roller-full-2048
run_train Mjlab-Velocity-Swizzle-MicroDuck swizzle-full-2048

# Roulade / ball kick: default max_iterations=10000.
run_train Mjlab-Roulade-Flat-MicroDuck roulade-full-2048
run_train Mjlab-BallKick-Flat-MicroDuck ball-kick-right-full-2048

echo "[$(date -Iseconds)] ALL DONE"
