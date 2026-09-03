#!/usr/bin/env bash
# Sequential skill training queue (2048 envs, default max_iterations per task).
# Order: VelStand-Rough → BallKick-Left → StandUp/SitStand/GroundPick → roller extras.
# Backlash variants intentionally omitted (train on real-robot gap only).
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl

NUM_ENVS=2048
LOG=/tmp/skill_train.log
BALL_KICK_CFG=src/mjlab_microduck/tasks/microduck_ball_kick_env_cfg.py

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

run_ball_kick_left() {
  if ! rg -q '^KICK_FOOT = "right"' "$BALL_KICK_CFG"; then
    echo "ERROR: expected KICK_FOOT = \"right\" in $BALL_KICK_CFG"
    exit 1
  fi
  sed -i 's/^KICK_FOOT = "right"/KICK_FOOT = "left"/' "$BALL_KICK_CFG"
  trap 'sed -i "s/^KICK_FOOT = \"left\"/KICK_FOOT = \"right\"/" "$BALL_KICK_CFG"' RETURN
  run_train Mjlab-BallKick-Flat-MicroDuck ball-kick-left-full-2048
}

echo "[$(date -Iseconds)] SKILL TRAIN QUEUE START"

# 1. Walking + fall recovery (closer to official alpha_walking role)
run_train Mjlab-VelStand-Rough-MicroDuck velstand-rough-full-2048

# 2. Left-foot kick (module flag flipped for this run only)
run_ball_kick_left

# 3. Daily skills (flat terrain; matches official policy roles)
run_train Mjlab-StandUp-Flat-MicroDuck standup-flat-full-2048
run_train Mjlab-SitStand-Flat-MicroDuck sitstand-flat-full-2048
run_train Mjlab-GroundPick-Flat-MicroDuck groundpick-flat-full-2048

# 4. Roller extras
run_train Mjlab-RollerCrouch-Flat-MicroDuck roller-crouch-full-2048
run_train Mjlab-Spin-Flat-MicroDuck spin-full-2048
run_train Mjlab-RollerSlope-Flat-MicroDuck roller-slope-full-2048
run_train Mjlab-RollerStandUp-Flat-MicroDuck roller-standup-full-2048

echo "[$(date -Iseconds)] SKILL TRAIN QUEUE ALL DONE"
