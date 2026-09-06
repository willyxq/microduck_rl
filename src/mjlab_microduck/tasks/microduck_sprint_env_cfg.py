"""Microduck SPRINT — barefoot flat-ground top speed.

Port of the DuckEMW / Hannes sprint recipe onto this repo's velocity stack
(61-D obs, BAM actuators, shared DR). Goal: push commanded forward speed from
the walking band (~0.4 m/s) toward ~2 m/s without collapsing into
park-and-farm or in-place high-stepping.

Key deltas vs ``make_microduck_velocity_env_cfg`` (see DuckEMW docs/03):

  - lin_vel_x curriculum 0.8 → 2.0 (stages keep tracking gradient alive)
  - forward-gated air_time + window curriculum (no in-place flight farm)
  - track_linear_velocity weight↑ / std loosened; loose floor term added
  - standing-env fraction capped at 5%; action_rate tax capped at -0.3
  - init_velocity_prob=0.5 (half the envs spawn already at commanded speed)
  - short 10 s episodes (more launch practice per GPU-hour)

Obs/action dims unchanged → ONNX hot-swaps into the walking runtime slot.
"""

from __future__ import annotations

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

# Command curriculum — end state (-0.2, 2.0); start inside reachable gradient.
LIN_VEL_X_START = (-0.1, 0.8)
LIN_VEL_X_FINAL = (-0.2, 2.0)
LIN_VEL_X_STAGES = [
    {"step": 0, "lin_vel_x": (-0.1, 0.8)},
    {"step": 750 * 24, "lin_vel_x": (-0.15, 1.2)},
    {"step": 1500 * 24, "lin_vel_x": (-0.2, 1.6)},
    {"step": 2500 * 24, "lin_vel_x": (-0.2, 2.0)},
]
LIN_VEL_Y_RANGE = (-0.1, 0.1)
ANG_VEL_Z_RANGE = (-0.5, 0.5)
REL_FORWARD_ENVS = 0.5
TURN_IN_PLACE_FRACTION = 0.0

AIR_TIME_MIN_START_S = 0.05
AIR_TIME_MIN_FINAL_S = 0.20
AIR_TIME_MAX_S = 0.45
AIR_TIME_WINDOW_STAGES = [
    {"step": 0, "threshold_min": AIR_TIME_MIN_START_S, "threshold_max": AIR_TIME_MAX_S},
    {"step": 500 * 24, "threshold_min": 0.10, "threshold_max": AIR_TIME_MAX_S},
    {"step": 1000 * 24, "threshold_min": 0.15, "threshold_max": AIR_TIME_MAX_S},
    {"step": 1500 * 24, "threshold_min": AIR_TIME_MIN_FINAL_S, "threshold_max": AIR_TIME_MAX_S},
]

STANDING_ENVS_FINAL = 0.05
ACTION_RATE_FINAL = -0.3
UPRIGHT_STD = math.sqrt(0.1)

TRACK_LIN_VEL_WEIGHT = 4.0
AIR_TIME_WEIGHT = 1.5
INIT_VELOCITY_PROB = 0.5
EPISODE_LENGTH_S = 10.0
TRACK_LIN_VEL_LOOSE_WEIGHT = 1.0
TRACK_LIN_VEL_LOOSE_STD = 1.0
TRACK_LIN_VEL_STD = math.sqrt(0.25)

NUM_STEPS_PER_ENV = 24


def make_microduck_sprint_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Sprint-specialised velocity env (flat). See module docstring."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = LIN_VEL_X_START
    command.ranges.lin_vel_y = LIN_VEL_Y_RANGE
    command.ranges.ang_vel_z = ANG_VEL_Z_RANGE
    command.rel_turn_in_place_envs = TURN_IN_PLACE_FRACTION
    command.rel_forward_envs = REL_FORWARD_ENVS
    command.init_velocity_prob = INIT_VELOCITY_PROB

    cfg.rewards["air_time"].func = microduck_mdp.feet_air_time_forward
    cfg.rewards["air_time"].weight = AIR_TIME_WEIGHT
    cfg.rewards["air_time"].params["threshold_min"] = AIR_TIME_MIN_START_S
    cfg.rewards["air_time"].params["threshold_max"] = AIR_TIME_MAX_S
    cfg.rewards["air_time"].params["command_threshold"] = 0.1

    cfg.curriculum["air_time_window"] = CurriculumTermCfg(
        func=microduck_mdp.air_time_window_curriculum,
        params={
            "reward_name": "air_time",
            "window_stages": AIR_TIME_WINDOW_STAGES,
        },
    )

    cfg.rewards["track_linear_velocity"].params["std"] = TRACK_LIN_VEL_STD
    cfg.rewards["track_linear_velocity"].weight = TRACK_LIN_VEL_WEIGHT

    cfg.curriculum["standing_envs"].params["standing_stages"] = [
        {"step": 0, "rel_standing_envs": 0.02},
        {"step": 500 * 24, "rel_standing_envs": 0.03},
        {"step": 1000 * 24, "rel_standing_envs": STANDING_ENVS_FINAL},
    ]

    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": -0.1},
        {"step": 500 * 24, "weight": -0.15},
        {"step": 1000 * 24, "weight": -0.2},
        {"step": 1500 * 24, "weight": ACTION_RATE_FINAL},
    ]

    cfg.rewards["upright"].params["std"] = UPRIGHT_STD

    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=velocity_mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": LIN_VEL_X_STAGES,
        },
    )

    # Loose tracking floor — gradient still visible at large speed errors.
    track_params = deepcopy(cfg.rewards["track_linear_velocity"].params)
    track_params["std"] = TRACK_LIN_VEL_LOOSE_STD
    cfg.rewards["track_linear_velocity_loose"] = RewardTermCfg(
        func=cfg.rewards["track_linear_velocity"].func,
        weight=TRACK_LIN_VEL_LOOSE_WEIGHT,
        params=track_params,
    )

    cfg.episode_length_s = EPISODE_LENGTH_S
    return cfg


MicroduckSprintRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="sprint",
    run_name="sprint",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    # DuckEMW default was 4k; extend toward the Hannes ~13.5k record run.
    max_iterations=10_000,
)
