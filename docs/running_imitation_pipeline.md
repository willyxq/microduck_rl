# Running gait imitation: DeepMimic and AMP

This experiment uses `HannesVonEssen/microduck-running` as an expert and runs
the complete pipeline locally in the free-base mjlab/MuJoCo-Warp/BAM simulator.
It is intentionally warm-started from the expert actor so a compact run tests
the imitation objectives rather than spending thousands of iterations
rediscovering standing and locomotion.

## Pipeline

1. `collect_running_reference.py` executes the trusted local expert checkpoint.
   It saves one canonical 4-second trajectory for DeepMimic and all collected
   state transitions for AMP.
2. DeepMimic uses PPO with explicit joint pose, joint velocity, root velocity,
   root angular velocity, orientation, contact, and action tracking rewards.
   Two unused command slots carry sine/cosine reference phase.
3. AMP uses PPO task rewards plus a learned transition discriminator. The
   discriminator sees the 48 physical observations from consecutive states,
   excluding all command slots.
4. `deepmimic_amp` warm-starts from a trained DeepMimic actor, then continues
   PPO with the task reward and AMP discriminator reward together.
5. `record_running_imitation.py` renders checkpoints in the same simulator and
   actuator model used for training. A simplified position-actuator MuJoCo
   scene is not dynamically equivalent and must not be used for this
   comparison.

## Reproduce

```bash
uv run python scripts/collect_running_reference.py \
  --checkpoint-file /path/to/HannesVonEssen/microduck-running/checkpoint.pt \
  --output-file artifacts/running_imitation/reference.npz

uv run python scripts/train_running_imitation.py \
  --mode deepmimic \
  --reference-file artifacts/running_imitation/reference.npz \
  --teacher-checkpoint /path/to/checkpoint.pt \
  --output-dir artifacts/running_imitation/deepmimic

uv run python scripts/train_running_imitation.py \
  --mode amp \
  --reference-file artifacts/running_imitation/reference.npz \
  --teacher-checkpoint /path/to/checkpoint.pt \
  --output-dir artifacts/running_imitation/amp

uv run python scripts/train_running_imitation.py \
  --mode deepmimic_amp \
  --reference-file artifacts/running_imitation/reference_expanded.npz \
  --initial-checkpoint artifacts/running_imitation/deepmimic/model_final.pt \
  --amp-reward-weight 0.1 --task-reward-weight 0.9 \
  --iterations 200 \
  --output-dir artifacts/running_imitation/deepmimic_amp_light
```

## Measured result on RTX 4080

| Policy | Training | Tracking reward | 1024-env validation speed | Survival |
|---|---:|---:|---:|---:|
| DeepMimic | 150 iterations, 314 s | 0.2896 | 1.052 m/s | 99.12% |
| AMP | 120 iterations, 253 s | — | 1.331 m/s | 90.14% |
| DeepMimic → AMP (0.35) | 300 iterations, 230 s | 0.2583 | 0.781 m/s | 96.78% |
| DeepMimic → AMP (0.10) | 200 iterations, 150 s | 0.2722 | 0.838 m/s | 96.48% |

The source expert measured about 1.64 m/s over a separate 64-environment
battery. In matched-physics videos, the selected deterministic seeds measured
1.57 m/s (expert), 0.84 m/s (DeepMimic), and 0.55 m/s (the lighter sequential
AMP run). The expanded dataset contains 102,400 transitions from 512 parallel
rollouts while preserving the original canonical trajectory for exact tracking.

These short runs prove that both learning loops and free-base deployment work;
they do not outperform the expert. DeepMimic is more stable but slower because
it follows one imperfect four-second rollout closely. AMP preserves a looser
running style and is more sensitive to random initial dynamics. On this
single-gait dataset, adding AMP after DeepMimic did not improve tracking reward,
speed, or survival; reducing AMP weight reduced, but did not remove, the
regression. More genuinely diverse expert clips, reference-state initialization,
longer training, and discriminator regularization are the next steps for
production quality.

## Video-to-dance experiment

`extract_video_dance_motion.py` uses YOLO pose landmarks from a source clip and
maps squat/lift rhythm, hip sway, and upper-body orientation to MicroDuck action
offsets. `collect_video_dance_reference.py` overlays those offsets on a stable
VelStand teacher, rejects failed rollouts, and saves both a canonical trajectory
and transition samples.

For the first 8.02 seconds of a Gangnam Style horse-dance clip, all 240 source
frames had a detected person (mean valid-keypoint confidence 0.966). Retargeting
at 50 Hz and 3× action scale yielded 401 frames and 205,312 expert transitions;
all 512 collection environments survived.

| Dance policy | Training | Tracking reward | Survival |
|---|---:|---:|---:|
| DeepMimic | 250 iterations, 183 s | 0.6713 | 100% |
| DeepMimic → AMP (0.25) | 300 iterations, 222 s | 0.5382 | 99.51% |

DeepMimic remained closest to the retargeted trajectory and most stable. The
sequential AMP stage produced larger squats, turns, and head/body motion, but
lowered exact tracking and robustness. MicroDuck has no arms, so the crossed-arm
horse-riding gesture cannot be represented; the retargeting preserves the
lower-body rhythm and torso/head style instead.
