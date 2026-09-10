# MicroDuck imitation-learning example

This example trains a policy from demonstrated joint trajectories instead of
defining a task-specific reinforcement-learning reward. The included
`nod_sway.json` demonstration is deliberately small and hand-authored so the
entire data-to-policy path can be inspected.

## How imitation learning produces richer motion

1. Capture motion (mocap, teleoperation, animation, or another controller).
2. Retarget it to MicroDuck's body proportions and 14 joints.
3. Pair each observed state with the demonstrated next action.
4. Train a policy to minimize behavior-cloning error
   `mean((policy(observation) - demonstrated_action)^2)`.
5. Deploy the resulting policy through the same 61D-observation to
   14D-action interface as the RL policies.

The motion detail is carried by the examples, so adding behaviors does not
require inventing a separate “bow reward”, “dance reward”, and so on.

The minimal example encodes cycle phase as sine/cosine in the first two command
slots and learns a 61D-to-14D MLP. Other observation fields are independently
noised during training, forcing the network to learn the demonstrated
phase-to-pose mapping rather than accidental synthetic correlations.

## Run it

```bash
uv run --with pytest pytest tests/test_imitation.py -q
uv run python scripts/train_imitation_bc.py \
  --motion examples/imitation_learning/nod_sway.json \
  --output-dir artifacts/imitation_nod_sway
uv run python scripts/record_imitation_demo.py \
  --policy artifacts/imitation_nod_sway/policy.onnx \
  --motion examples/imitation_learning/nod_sway.json \
  --output artifacts/imitation_nod_sway/demo.mp4
```

Replace the keyframes in the JSON with a retargeted trajectory to teach a
different cyclic motion.

The renderer uses a virtual root gantry by default. This intentionally isolates
“did the network learn the demonstrated joint motion?” from the separate,
harder balance problem. Pass `--free-base` to expose the plain BC policy to
full-body dynamics; this tiny phase-only example is expected to fall.

## What this example does not solve

Plain behavior cloning sees only states in the demonstrations. After a push or
modeling error, it can visit unfamiliar states and compound its mistakes. A
production motion-imitation system normally adds:

- many retargeted clips and randomized initial phases;
- reference pose, velocity, root, and contact tracking rewards (DeepMimic);
- perturbations, recovery curriculum, and dynamics randomization;
- optionally an adversarial motion prior (AMP/GAIL) or a diffusion policy.

Those methods retain the reusable motion-tracking objective while using RL or
generative modeling to improve robustness and multimodality. Therefore this
demo proves the imitation data and deployment path under a virtual gantry; it
is not a claim of free-base balance, push-robust locomotion, or human-mocap
retargeting.
