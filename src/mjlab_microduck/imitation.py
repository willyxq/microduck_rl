"""Minimal trajectory-imitation utilities for MicroDuck.

The motion file stores demonstrated joint offsets from HOME.  A behavior
cloning policy learns the deployment contract used by MicroDuck:
61 observations -> 14 joint-position offsets.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

OBS_DIM = 61
ACTION_DIM = 14
COMMAND_START = 48
PHASE_SIN_INDEX = COMMAND_START
PHASE_COS_INDEX = COMMAND_START + 1

JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

# HOME_FRAME in actuator order. Actions are offsets from this pose.
DEFAULT_POSE = np.array(
    [
        0.0,
        -0.0873,
        -0.4579,
        -0.0049,
        0.4530,
        0.3491,
        0.3491,
        0.0,
        0.0,
        0.0,
        0.0873,
        0.4579,
        0.0049,
        -0.4530,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Motion:
    name: str
    period_s: float
    phases: np.ndarray
    actions: np.ndarray


def load_motion(path: str | Path) -> Motion:
    """Load and validate a periodic demonstrated motion."""
    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream)

    period_s = float(raw["period_s"])
    if not math.isfinite(period_s) or period_s <= 0:
        raise ValueError("period_s must be positive and finite")
    if tuple(raw["joint_order"]) != JOINT_NAMES:
        raise ValueError("joint_order does not match MicroDuck actuator order")

    keyframes = raw["keyframes"]
    phases = np.asarray([frame["phase"] for frame in keyframes], dtype=np.float32)
    actions = np.asarray([frame["action"] for frame in keyframes], dtype=np.float32)
    if phases.ndim != 1 or actions.shape != (len(phases), ACTION_DIM):
        raise ValueError(f"keyframes must contain {ACTION_DIM}-element actions")
    if len(phases) < 2 or phases[0] != 0.0 or phases[-1] != 1.0:
        raise ValueError("keyframe phases must start at 0 and end at 1")
    if np.any(np.diff(phases) <= 0):
        raise ValueError("keyframe phases must be strictly increasing")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions must be finite")
    if not np.allclose(actions[0], actions[-1], atol=1e-7):
        raise ValueError("periodic motion must have equal first and last actions")
    return Motion(str(raw["name"]), period_s, phases, actions)


def interpolate_actions(motion: Motion, phases: np.ndarray) -> np.ndarray:
    """Interpolate keyframes with smoothstep easing and periodic phase wrap."""
    wrapped = np.mod(np.asarray(phases, dtype=np.float32), 1.0)
    segment = np.searchsorted(motion.phases, wrapped, side="right") - 1
    segment = np.clip(segment, 0, len(motion.phases) - 2)
    left_phase = motion.phases[segment]
    right_phase = motion.phases[segment + 1]
    t = (wrapped - left_phase) / (right_phase - left_phase)
    t = t * t * (3.0 - 2.0 * t)
    return (
        motion.actions[segment] * (1.0 - t[:, None])
        + motion.actions[segment + 1] * t[:, None]
    ).astype(np.float32)


def build_dataset(
    motion: Motion,
    sample_count: int,
    seed: int = 0,
    observation_noise: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create BC samples.

    Non-command observations are randomized independently.  This prevents the
    tiny example from relying on synthetic state correlations: phase alone
    selects the demonstrated action.
    """
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 1.0, sample_count).astype(np.float32)
    observations = rng.normal(0.0, observation_noise, (sample_count, OBS_DIM)).astype(
        np.float32
    )
    observations[:, 5] -= 1.0  # projected gravity near upright
    observations[:, PHASE_SIN_INDEX] = np.sin(2.0 * np.pi * phases)
    observations[:, PHASE_COS_INDEX] = np.cos(2.0 * np.pi * phases)
    observations[:, COMMAND_START + 2 :] = 0.0
    actions = interpolate_actions(motion, phases)
    return torch.from_numpy(observations), torch.from_numpy(actions)


class BehaviorCloningPolicy(nn.Module):
    """Small MLP with the same I/O dimensions as deployed MicroDuck policies."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(OBS_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


def phase_observation(phase: float) -> np.ndarray:
    """Create a nominal upright observation for model sanity checks."""
    observation = np.zeros(OBS_DIM, dtype=np.float32)
    observation[5] = -1.0
    observation[PHASE_SIN_INDEX] = math.sin(2.0 * math.pi * phase)
    observation[PHASE_COS_INDEX] = math.cos(2.0 * math.pi * phase)
    return observation
