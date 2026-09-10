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
4. `record_running_imitation.py` renders checkpoints in the same simulator and
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
```

## Measured result on RTX 4080

| Policy | Training | 1024-env validation speed | Survival |
|---|---:|---:|---:|
| DeepMimic | 150 iterations, 314 s | 1.052 m/s | 99.12% |
| AMP | 120 iterations, 253 s | 1.331 m/s | 90.14% |

The source expert measured about 1.64 m/s over a separate 64-environment
battery. In matched-physics videos, the selected deterministic seeds measured
1.57 m/s (expert), 0.84 m/s (DeepMimic), and 0.89 m/s (AMP).

These short runs prove that both learning loops and free-base deployment work;
they do not outperform the expert. DeepMimic is more stable but slower because
it follows one imperfect four-second rollout closely. AMP preserves a looser
running style and is more sensitive to random initial dynamics. More diverse
expert clips, reference-state initialization, longer training, and discriminator
regularization are the next steps for production quality.
