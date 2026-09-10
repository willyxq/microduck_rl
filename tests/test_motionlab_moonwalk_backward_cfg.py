from mjlab_microduck.tasks.microduck_motionlab_moonwalk_backward_env_cfg import (
    GeneratedRlCfg,
    make_generated_env_cfg,
)


def test_generated_velocity_cfg_contract():
    cfg = make_generated_env_cfg(play=False, rough=False)
    assert "twist" in cfg.commands
    assert "track_linear_velocity" in cfg.rewards
    assert tuple(cfg.commands["twist"].ranges.lin_vel_x) == (-0.25, -0.08)
    assert tuple(cfg.commands["twist"].ranges.lin_vel_y) == (-0.03, 0.03)
    assert tuple(cfg.commands["twist"].ranges.ang_vel_z) == (-0.15, 0.15)
    assert cfg.commands["twist"].rel_standing_envs == 0.08
    assert cfg.rewards["foot_slip"].weight == -0.015
    assert GeneratedRlCfg.algorithm.symmetry_cfg is not None
