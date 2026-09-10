"""Exact task overlay published by the upstream Moonwalk Motion Lab artifact."""

from copy import deepcopy

from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG


def make_generated_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    command = cfg.commands["twist"]
    command.ranges.lin_vel_x = (-0.25, -0.08)
    command.ranges.lin_vel_y = (-0.03, 0.03)
    command.ranges.ang_vel_z = (-0.15, 0.15)
    command.rel_standing_envs = 0.08
    command.rel_turn_in_place_envs = 0.0
    cfg.rewards["foot_slip"].weight = -0.015
    return cfg


GeneratedRlCfg = deepcopy(MicroduckRlCfg)
GeneratedRlCfg.experiment_name = "motionlab_moonwalk_backward_official_repro"
GeneratedRlCfg.run_name = "moonwalk_backward_official_repro"
GeneratedRlCfg.algorithm.symmetry_cfg = SYMMETRY_CFG
