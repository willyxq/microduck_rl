#!/usr/bin/env bash
# Sequential BeyondMimic training for the remaining UMR clips.
set -u
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
LOG=logs/tracking_remaining_train.log
mkdir -p logs
: > "$LOG"

tasks=(
  Mjlab-Tracking-Flat-MicroDuck-ZombieWalk
  Mjlab-Tracking-Flat-MicroDuck-Run
  Mjlab-Tracking-Flat-MicroDuck-Jump
  Mjlab-Tracking-Flat-MicroDuck-Dance
  Mjlab-Tracking-Flat-MicroDuck-Spinkick
)

for task in "${tasks[@]}"; do
  echo "======== START $task $(date '+%F %T') ========" | tee -a "$LOG"
  if uv run train "$task" --env.scene.num-envs 4096 >>"$LOG" 2>&1; then
    echo "======== OK $task $(date '+%F %T') ========" | tee -a "$LOG"
  else
    echo "======== FAIL $task exit=$? $(date '+%F %T') ========" | tee -a "$LOG"
  fi
done
echo "======== ALL DONE $(date '+%F %T') ========" | tee -a "$LOG"
