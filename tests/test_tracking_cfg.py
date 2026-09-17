from pathlib import Path

import mujoco
import numpy as np

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML
from mjlab_microduck.tasks.microduck_tracking_env_cfg import (
    ANCHOR_POS_FAIL_M,
    EE_POS_FAIL_M,
    POS_REWARD_STD_M,
    TRACKING_BODY_NAMES,
    TRACKING_EE_BODY_NAMES,
    UMR_WALK_MOTION_FILE,
    make_microduck_tracking_env_cfg,
    MicroduckTrackingRlCfg,
)


def test_walk_motion_npz_has_beyondmimic_keys():
    assert UMR_WALK_MOTION_FILE.exists(), UMR_WALK_MOTION_FILE
    data = np.load(UMR_WALK_MOTION_FILE, allow_pickle=True)
    for key in (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        assert key in data.files, key
    assert float(np.asarray(data["fps"]).reshape(-1)[0]) == 50.0
    t = data["joint_pos"].shape[0]
    assert t >= 40
    assert data["joint_pos"].shape[1] == 14
    assert data["body_pos_w"].shape[0] == t
    assert data["body_pos_w"].shape[-1] == 3
    assert data["body_quat_w"].shape[-1] == 4


def test_tracking_bodies_exist_on_walk_model():
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_WALK_XML))
    names = {model.body(i).name for i in range(model.nbody)}
    missing = [n for n in TRACKING_BODY_NAMES + TRACKING_EE_BODY_NAMES if n not in names]
    assert not missing, missing
    assert TRACKING_BODY_NAMES[0] == "trunk_base"


def test_tracking_cfg_is_duck_scaled_beyondmimic():
    cfg = make_microduck_tracking_env_cfg()
    motion = cfg.commands["motion"]
    assert Path(motion.motion_file) == UMR_WALK_MOTION_FILE
    assert motion.anchor_body_name == "trunk_base"
    assert motion.body_names[0] == "trunk_base"
    assert cfg.actions["joint_pos"].scale == 1.0
    assert cfg.rewards["motion_global_root_pos"].params["std"] == POS_REWARD_STD_M
    assert cfg.rewards["motion_body_pos"].params["std"] == POS_REWARD_STD_M
    assert cfg.terminations["anchor_pos"].params["threshold"] == ANCHOR_POS_FAIL_M
    assert cfg.terminations["ee_body_pos"].params["threshold"] == EE_POS_FAIL_M
    assert cfg.terminations["ee_body_pos"].params["body_names"] == TRACKING_EE_BODY_NAMES
    assert "nan_state" in cfg.terminations
    assert "expand_bam_friction_fields" in cfg.events
    com_ranges = cfg.events["base_com"].params["ranges"]
    assert max(abs(v) for pair in com_ranges.values() for v in pair) <= 0.005
    assert cfg.viewer.body_name == "trunk_base"
    assert MicroduckTrackingRlCfg.experiment_name == "tracking_walk"


def test_play_cfg_disables_rsi_and_pushes():
    cfg = make_microduck_tracking_env_cfg(play=True)
    motion = cfg.commands["motion"]
    assert motion.sampling_mode == "start"
    assert motion.pose_range == {}
    assert motion.velocity_range == {}
    assert "push_robot" not in cfg.events
    assert cfg.observations["actor"].enable_corruption is False
