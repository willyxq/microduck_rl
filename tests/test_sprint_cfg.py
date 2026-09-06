"""Sprint env cfg invariants — locks the sprint recipe against regressions."""

import math

from mjlab_microduck.tasks.microduck_sprint_env_cfg import (
    AIR_TIME_MAX_S,
    AIR_TIME_MIN_FINAL_S,
    AIR_TIME_MIN_START_S,
    AIR_TIME_WINDOW_STAGES,
    make_microduck_sprint_env_cfg,
    MicroduckSprintRlCfg,
)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def test_sprint_command_ranges():
    cfg = make_microduck_sprint_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.ranges.lin_vel_x == (-0.1, 0.8)
    assert cmd.ranges.lin_vel_y == (-0.1, 0.1)
    assert cmd.ranges.ang_vel_z == (-0.5, 0.5)
    assert cmd.rel_turn_in_place_envs == 0.0
    assert cmd.rel_forward_envs == 0.5


def test_air_time_window_curriculum_ratchets_up_to_flight_phase():
    sprint = make_microduck_sprint_env_cfg()
    params = sprint.rewards["air_time"].params
    assert (params["threshold_min"], params["threshold_max"]) == (
        AIR_TIME_MIN_START_S,
        AIR_TIME_MAX_S,
    )
    stages = sprint.curriculum["air_time_window"].params["window_stages"]
    assert stages == AIR_TIME_WINDOW_STAGES
    assert stages[-1]["threshold_min"] == AIR_TIME_MIN_FINAL_S
    assert stages[-1]["threshold_max"] == AIR_TIME_MAX_S
    mins = [st["threshold_min"] for st in stages]
    assert mins == sorted(mins)
    assert sprint.rewards["air_time"].weight == 1.5
    assert params["command_threshold"] == 0.1


def test_tracking_std_keeps_gradient_at_sprint_speed():
    cfg = make_microduck_sprint_env_cfg()
    std = cfg.rewards["track_linear_velocity"].params["std"]
    assert std == math.sqrt(0.25)
    assert math.exp(-(1.0 / std) ** 2) > 0.01
    assert cfg.rewards["track_linear_velocity"].weight == 4.0


def test_anti_violence_regularizers_inherited():
    cfg = make_microduck_sprint_env_cfg()
    assert cfg.rewards["action_rate_l2"].weight == -0.1
    stages = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert stages[-1]["weight"] == -0.3
    weights = [st["weight"] for st in stages]
    assert weights == sorted(weights, reverse=True)
    assert cfg.rewards["body_ang_vel"].weight < 0
    assert cfg.rewards["angular_momentum"].weight < 0
    assert cfg.rewards["self_collisions"].weight < 0
    assert cfg.rewards["foot_slip"].weight < 0
    assert cfg.rewards["dof_pos_limits"].weight < 0


def test_command_speed_curriculum_reaches_2ms():
    cfg = make_microduck_sprint_env_cfg()
    stages = cfg.curriculum["command_vel"].params["velocity_stages"]
    xmax = [st["lin_vel_x"][1] for st in stages]
    assert xmax == [0.8, 1.2, 1.6, 2.0]
    assert stages[-1]["lin_vel_x"] == (-0.2, 2.0)


def test_loose_tracking_floor_term():
    cfg = make_microduck_sprint_env_cfg()
    term = cfg.rewards["track_linear_velocity_loose"]
    assert term.weight == 1.0
    assert term.params["std"] == 1.0
    assert term.params["command_name"] == "twist"
    assert math.exp(-((2.0 / 1.0) ** 2)) > 0.01


def test_burst_training_setup():
    cfg = make_microduck_sprint_env_cfg()
    assert cfg.commands["twist"].init_velocity_prob == 0.5
    assert cfg.episode_length_s == 10.0


def test_air_time_is_forward_gated():
    from mjlab_microduck.tasks import mdp as microduck_mdp

    cfg = make_microduck_sprint_env_cfg()
    assert cfg.rewards["air_time"].func is microduck_mdp.feet_air_time_forward
    assert cfg.commands["twist"].init_velocity_prob == 0.5


def test_upright_std_allows_sprint_lean():
    cfg = make_microduck_sprint_env_cfg()
    assert cfg.rewards["upright"].params["std"] == math.sqrt(0.1)
    assert cfg.rewards["upright"].weight == 2.0


def test_bam_and_nan_guard_invariants_kept():
    cfg = make_microduck_sprint_env_cfg()
    assert "expand_bam_friction_fields" in cfg.events
    assert cfg.events["expand_bam_friction_fields"].mode == "startup"
    assert "nan_state" in cfg.terminations


def test_actor_observation_keeps_the_61d_slot_layout():
    cfg = make_microduck_sprint_env_cfg()
    terms = cfg.observations["actor"].terms
    assert "base_lin_vel" not in terms
    assert "height_scan" not in terms
    assert list(terms.keys()) == [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
        "head_command",
        "body_command",
    ]


def test_obs_parity_with_velocity_env():
    sprint = make_microduck_sprint_env_cfg()
    walk = make_microduck_velocity_env_cfg()
    for grp in ("actor", "critic"):
        assert list(sprint.observations[grp].terms.keys()) == list(
            walk.observations[grp].terms.keys()
        ), f"obs layout diverges on group {grp}"


def test_zero_command_behavior_still_trained():
    cfg = make_microduck_sprint_env_cfg()
    stages = cfg.curriculum["standing_envs"].params["standing_stages"]
    assert stages[0]["rel_standing_envs"] > 0.0
    assert stages[-1]["rel_standing_envs"] == 0.05


def test_runner_cfg_is_sprint_specific():
    assert MicroduckSprintRlCfg.experiment_name == "sprint"
    assert MicroduckSprintRlCfg.run_name == "sprint"
    assert MicroduckSprintRlCfg.max_iterations == 10_000
    assert MicroduckSprintRlCfg.actor.obs_normalization is True
