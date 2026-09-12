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

### Direct MicroDuck video tracking

For a later Korean duck-dance clip that shows a person on the left and a
physical MicroDuck on the right, `extract_duck_video_motion.py` tracks the robot
itself with CoTracker3 instead of applying human proportions to it. Fourteen
visual anchors cover the head, neck, pelvis, hips, knees, shins, and feet. Their
relative 2D motion drives mirrored hip-pitch, knee, ankle, hip-roll, and head
offsets.

The robot turns side-on after the first three seconds, causing persistent
self-occlusion. Only the 0.5–3.5 s frontal section was accepted: mean point
visibility was 91.1%, and the least visible anchor still covered 68.9% of
frames. The retargeted 3 s cycle produced 76,800 expert transitions from 512
randomized environments with 100% one-cycle survival.

| Direct-video policy | Training | Tracking reward | Survival |
|---|---:|---:|---:|
| DeepMimic | 300 iterations, 221 s | 0.7731 | 100% |
| DeepMimic → AMP (0.25) | 300 iterations, 226 s | 0.6497 | 99.71% |

Direct robot tracking improved DeepMimic tracking over the earlier human-only
Gangnam retarget (0.7731 versus 0.6713). AMP again increased visible variation
and lean, while reducing trajectory fidelity and a small amount of robustness.
This remains monocular 2D retargeting: occluded joints and out-of-plane angles
cannot be recovered as accurately as encoder logs, motion capture, or a
calibrated multi-view recording.

The first teacher did not look like the filmed duck. CoTracker queries sat on
the Bilibili caption, so the feet never moved (≤1 px), and the 90° turn is
whole-body root yaw that a 14-DoF VelStand overlay cannot show. Peak action
offset was only 0.12 rad.

`extract_filmed_duck_pose.py` instead crops the right-panel duck, anchors on
the orange bill / camera-eye / feet, and compares two reconstructions:

| Teacher | Peak offset | Head yaw | Root yaw | Notes |
|---|---:|---:|---:|---|
| Old CoTracker + VelStand | 0.12 rad | ~0.15 rad | 0 | standing wiggle |
| Appearance + analytic | 0.98 rad | 0.97 rad | 1.55 rad | selected |
| Appearance + MuJoCo IK | 1.20 rad | 1.34 rad | 1.96 rad | overfits noisy feet |

The selected teacher is a kinematic gantry replay (`record_kinematic_teacher.py`)
of the analytic joints plus root yaw. That is the pose the later DeepMimic
stage should track.

### Human duck-step (Bilibili BV128CqYFEcR)

`extract_duck_step_motion.py` retargets a human-only duck walk onto hip and
knee offsets. MicroDuck has no independent waist or spine: `trunk_base` is a
rigid free body. Hip yaw is about ±25° / ±30°, hip roll about ±22°, and hip
pitch about ±90°. A human waist twist or lean can only be approximated:

- opposite-sign hip yaw for the missing waist twist / duck-toed waddle
- same-sign hip roll for weight shift
- hip pitch + knee + ankle for the squat and low step

The usable dance is in 8–16 s of the source clip (240 frames, 100% detection,
mean keypoint confidence 0.963). At 50 Hz this is a 400-frame, 8.0 s cycle
with a 0.462 rad peak action offset, 0.324 rad hip-yaw range, and 0.256 rad
hip-roll range. Overlaying those offsets on VelStand at motion scale 1.4
yielded 120,400 transitions from 301 of 512 collection environments (58.8%
one-cycle survival). Scale 1.0 survived completely but looked conservative;
scale 2.0 collapsed to 2% survival.

| Duck-step policy | Training | Tracking reward | Survival |
|---|---:|---:|---:|
| DeepMimic | 300 iterations, 216 s | 0.7250 | 99.80% |
| DeepMimic → AMP (0.25) | 300 iterations, 225 s | 0.5459 | 97.56% |

DeepMimic stayed closer to the retargeted waddle and more upright (render
tilt 5.9°). Sequential AMP increased lean and variation (render tilt 9.1°)
while lowering exact tracking. The remaining visual gap versus the human
clip is mostly the missing waist DOF and monocular 2D depth, not a failed
hip-yaw/roll mapping.
