"""BeyondMimic tracking on MicroDuck.

mjlab's tracking task is a reimplementation of BeyondMimic
(https://beyondmimic.github.io/). This config swaps in the walk-model duck,
points ``MotionCommand`` at a local UMR→mjlab ``motion.npz``, and scales the
G1-sized residuals / RSI / CoM ranges down to a ~27 cm robot.

This is *not* the 61D velocity-family obs contract. The actor sees the
BeyondMimic command (joint pos/vel) plus anchor residuals; export goes
through ``MotionTrackingOnPolicyRunner`` (obs + time_step).
"""

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp

UMR_WALK_MOTION_FILE = (
    Path(__file__).resolve().parents[1] / "motions" / "umr_humanoid_walk.npz"
)

# Free-root first: MotionCommand writes body_names[0] as the reset root.
# trunk_base is also the BeyondMimic anchor (G1 uses pelvis + torso_link;
# the duck's free joint *is* the torso).
TRACKING_BODY_NAMES = (
    "trunk_base",
    "yaw2roll",
    "hip_l",
    "upper_leg_left",
    "leg",
    "ankle_left",
    "bearing_roll",
    "hip_l_2",
    "upper_leg_right",
    "leg_2",
    "ankle_right",
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "jaw_soft",
)

TRACKING_EE_BODY_NAMES = (
    "ankle_left",
    "ankle_right",
    "jaw_soft",
)

# Position Gaussians / fail-z are in metres. G1 defaults (0.3 / 0.25) are
# about one duck-height and give a flat reward on a 27 cm robot.
POS_REWARD_STD_M = 0.05
ORI_REWARD_STD = 0.4
LIN_VEL_REWARD_STD = 0.4
ANG_VEL_REWARD_STD = 3.14
ANCHOR_POS_FAIL_M = 0.06
EE_POS_FAIL_M = 0.06

DUCK_POSE_RANGE = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (-0.005, 0.005),
    "roll": (-0.1, 0.1),
    "pitch": (-0.1, 0.1),
    "yaw": (-0.2, 0.2),
}

DUCK_VELOCITY_RANGE = {
    "x": (-0.15, 0.15),
    "y": (-0.15, 0.15),
    "z": (-0.08, 0.08),
    "roll": (-0.3, 0.3),
    "pitch": (-0.3, 0.3),
    "yaw": (-0.4, 0.4),
}


def make_microduck_tracking_env_cfg(
    play: bool = False,
    motion_file: str | Path | None = None,
) -> ManagerBasedRlEnvCfg:
    """Create MicroDuck flat-terrain BeyondMimic tracking configuration."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = str(motion_file or UMR_WALK_MOTION_FILE)
    motion_cmd.anchor_body_name = "trunk_base"
    motion_cmd.body_names = TRACKING_BODY_NAMES
    motion_cmd.pose_range = dict(DUCK_POSE_RANGE)
    motion_cmd.velocity_range = dict(DUCK_VELOCITY_RANGE)

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
        r"^(left|right)_foot_collision$"
    )
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)
    cfg.events["base_com"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.events["base_com"].params["ranges"] = {
        0: (-0.003, 0.003),
        1: (-0.003, 0.003),
        2: (-0.003, 0.003),
    }
    cfg.events["push_robot"].params["velocity_range"] = dict(DUCK_VELOCITY_RANGE)

    # BAM writes per-env dof_frictionloss/dof_damping; mjlab only expands
    # fields declared by startup events.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    cfg.rewards["motion_global_root_pos"].params["std"] = POS_REWARD_STD_M
    cfg.rewards["motion_body_pos"].params["std"] = POS_REWARD_STD_M
    cfg.rewards["motion_global_root_ori"].params["std"] = ORI_REWARD_STD
    cfg.rewards["motion_body_ori"].params["std"] = ORI_REWARD_STD
    cfg.rewards["motion_body_lin_vel"].params["std"] = LIN_VEL_REWARD_STD
    cfg.rewards["motion_body_ang_vel"].params["std"] = ANG_VEL_REWARD_STD
    cfg.rewards["self_collisions"].params["force_threshold"] = 2.0

    cfg.terminations["anchor_pos"].params["threshold"] = ANCHOR_POS_FAIL_M
    cfg.terminations["ee_body_pos"].params["threshold"] = EE_POS_FAIL_M
    cfg.terminations["ee_body_pos"].params["body_names"] = TRACKING_EE_BODY_NAMES
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (self_collision_cfg.name,)},
    )

    cfg.viewer.body_name = "trunk_base"
    cfg.viewer.distance = 0.55

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.sampling_mode = "start"

    return cfg


MicroduckTrackingRlCfg = RslRlOnPolicyRunnerCfg(
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
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="tracking_walk",
    run_name="umr_walk",
    wandb_tags=("beyondmimic", "umr", "walk"),
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=10_000,
)
