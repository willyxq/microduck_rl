#!/usr/bin/env bash
# Film official pollen-robotics/microduck-policies ONNX (local copy) and
# matching locally-trained ONNX for side-by-side comparison.
set -euo pipefail
ROOT=/home/william/Workspace/e1901/microduck
REPO="$ROOT/microduck_deepmimic"
OFF="$ROOT/microduck/policies"
OUT="$ROOT/exports/hf_replay"
LOC="$ROOT/exports/local_replay"
export UV_PROJECT_ENVIRONMENT="$ROOT/microduck_rl/.venv"
export MUJOCO_GL=egl
mkdir -p "$OUT" "$LOC"
cd "$REPO"

rec() {
  echo "=== $* ==="
  uv run python scripts/record_hf_onnx_demo.py "$@"
}

# Official HF pack
rec --onnx "$OFF/alpha_walking.onnx" --kind walking --vx 0.35 --seconds 6 \
  --output "$OUT/official_alpha_walking.mp4"
rec --onnx "$OFF/alpha_stand.onnx" --kind standing --seconds 4 \
  --output "$OUT/official_alpha_stand.mp4"
rec --onnx "$OFF/alpha_sitstand.onnx" --kind sitstand --seconds 8 --trigger-at 1.0 \
  --output "$OUT/official_alpha_sitstand.mp4"
rec --onnx "$OFF/alpha_ground_pick.onnx" --kind ground_pick --seconds 6 --trigger-at 1.0 \
  --stand-onnx "$OFF/alpha_stand.onnx" --output "$OUT/official_alpha_ground_pick.mp4"
rec --onnx "$OFF/roulade.onnx" --kind roulade --seconds 6 --trigger-at 1.2 \
  --stand-onnx "$OFF/alpha_stand.onnx" --output "$OUT/official_roulade.mp4"
rec --onnx "$OFF/ball_kick_right.onnx" --kind kick_right --scene ball --seconds 6 --trigger-at 1.2 \
  --stand-onnx "$OFF/alpha_stand.onnx" --output "$OUT/official_ball_kick_right.mp4"
rec --onnx "$OFF/ball_kick_left.onnx" --kind kick_left --scene ball --seconds 6 --trigger-at 1.2 \
  --stand-onnx "$OFF/alpha_stand.onnx" --output "$OUT/official_ball_kick_left.mp4"
rec --onnx "$OFF/roller.onnx" --kind walking --scene rollers --vx 0.4 --seconds 6 \
  --action-scale 0.8 --output "$OUT/official_roller.mp4"
rec --onnx "$OFF/roller_crouch.onnx" --kind walking --scene rollers --seconds 5 \
  --action-scale 0.8 --output "$OUT/official_roller_crouch.mp4"

# Hannes running ONNX on the same flat recorder (high cmd)
rec --onnx /home/william/Workspace/e1901/repro/hf/microduck-running/policy.onnx \
  --kind walking --vx 2.2 --seconds 6 --output "$OUT/hannes_running_onnx.mp4"

# Local trained counterparts (wandb-exported ONNX)
W="$ROOT/microduck_rl/wandb"
rec --onnx "$W/run-20260831_125748-979abg1b/files/2026-08-31_12-57-39_walk-full-2048.onnx" \
  --kind walking --vx 0.35 --seconds 6 --output "$LOC/local_walk.mp4"
rec --onnx "$W/run-20260904_175954-ylefmze0/files/2026-09-04_17-59-51_standup-flat-full-2048.onnx" \
  --kind standing --seconds 4 --output "$LOC/local_stand.mp4"
rec --onnx "$W/run-20260904_220444-o7nxqap5/files/2026-09-04_22-04-41_sitstand-flat-full-2048.onnx" \
  --kind sitstand --seconds 8 --trigger-at 1.0 --output "$LOC/local_sitstand.mp4"
rec --onnx "$W/run-20260905_043520-jzjggbg6/files/2026-09-05_04-35-16_groundpick-flat-full-2048.onnx" \
  --kind ground_pick --seconds 6 --trigger-at 1.0 --stand-onnx "$OFF/alpha_stand.onnx" \
  --output "$LOC/local_ground_pick.mp4"
rec --onnx "$W/run-20260903_150123-15kpf2bl/files/2026-09-03_15-01-19_roulade-full-2048.onnx" \
  --kind roulade --seconds 6 --trigger-at 1.2 --stand-onnx "$OFF/alpha_stand.onnx" \
  --output "$LOC/local_roulade.mp4"
rec --onnx "$W/run-20260903_180227-2606ywd9/files/2026-09-03_18-02-24_ball-kick-right-full-2048.onnx" \
  --kind kick_right --scene ball --seconds 6 --trigger-at 1.2 --stand-onnx "$OFF/alpha_stand.onnx" \
  --output "$LOC/local_kick_right.mp4"
rec --onnx "$W/run-20260904_144707-7p0qqgzr/files/2026-09-04_14-47-03_ball-kick-left-full-2048.onnx" \
  --kind kick_left --scene ball --seconds 6 --trigger-at 1.2 --stand-onnx "$OFF/alpha_stand.onnx" \
  --output "$LOC/local_kick_left.mp4"
rec --onnx "$W/run-20260902_130514-1mg4gvio/files/2026-09-02_13-05-08_roller-full-2048.onnx" \
  --kind walking --scene rollers --vx 0.4 --seconds 6 --action-scale 0.8 \
  --output "$LOC/local_roller.mp4"
rec --onnx "$W/run-20260905_094153-n2cyl7tg/files/2026-09-05_09-41-49_roller-crouch-full-2048.onnx" \
  --kind walking --scene rollers --seconds 5 --action-scale 0.8 \
  --output "$LOC/local_roller_crouch.mp4"
rec --onnx "$W/run-20260909_154021-d3m416qk/files/2026-09-09_15-40-17_running-max-speed.onnx" \
  --kind walking --vx 2.2 --seconds 6 --output "$LOC/local_running.mp4"

echo DONE
ls -lh "$OUT" "$LOC"
