#!/usr/bin/env bash
# Render play clips for every BeyondMimic tracking task that has a checkpoint.
set -u
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
EXP=/home/william/Workspace/e1901/microduck/exports

declare -A TASKS=(
  [walk]=Mjlab-Tracking-Flat-MicroDuck
  [zombie_walk]=Mjlab-Tracking-Flat-MicroDuck-ZombieWalk
  [run]=Mjlab-Tracking-Flat-MicroDuck-Run
  [jump]=Mjlab-Tracking-Flat-MicroDuck-Jump
  [dance]=Mjlab-Tracking-Flat-MicroDuck-Dance
  [spinkick]=Mjlab-Tracking-Flat-MicroDuck-Spinkick
)

latest_ckpt() {
  local key="$1"
  local dir
  dir=$(ls -1dt logs/rsl_rl/tracking_"${key}"/* 2>/dev/null | head -1) || true
  if [ -z "${dir:-}" ]; then
    return 1
  fi
  ls -1t "$dir"/model_*.pt 2>/dev/null | head -1
}

for key in walk zombie_walk run jump dance spinkick; do
  ckpt=$(latest_ckpt "$key" || true)
  if [ -z "${ckpt:-}" ]; then
    echo "SKIP $key (no checkpoint)"
    continue
  fi
  echo "RENDER $key $ckpt"
  uv run python scripts/render_checkpoint.py \
    --task "${TASKS[$key]}" \
    --checkpoint-file "$ckpt" \
    --out-dir "/tmp/beyondmimic_${key}_demo" \
    --duration-s 8 \
    --width 1280 \
    --height 720 \
    --distance 0.55 \
    --azimuth 120 \
    --elevation -8 \
    --follow-entity robot \
    --follow-dz 0.02 \
    --track-body trunk_base \
    --tag clip
  src=$(ls -1t /tmp/beyondmimic_${key}_demo/clip/rl-video-step-0.mp4 2>/dev/null | head -1)
  if [ -z "${src:-}" ]; then
    echo "FAIL no mp4 for $key"
    continue
  fi
  ffmpeg -y -i "$src" -an -vf fps=30 -c:v libx264 -pix_fmt yuv420p -crf 28 -preset fast -movflags +faststart \
    "$EXP/beyondmimic_${key}_demo.mp4" </dev/null
  ffmpeg -y -ss 2.5 -i "$EXP/beyondmimic_${key}_demo.mp4" -frames:v 1 -q:v 4 \
    "$EXP/beyondmimic_${key}_demo_poster.jpg" </dev/null
  ls -lh "$EXP/beyondmimic_${key}_demo.mp4"
done
