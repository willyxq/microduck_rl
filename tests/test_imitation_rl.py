import numpy as np
import pytest
import torch
from tensordict import TensorDict

from mjlab_microduck.imitation_rl import (
    HEAD_COMMAND_START,
    PHYSICAL_OBS_DIM,
    RunningReferenceWrapper,
    transition_features,
)


def test_amp_transition_features_exclude_commands():
    obs = torch.arange(61, dtype=torch.float32).reshape(1, 61)
    nxt = obs + 100
    features = transition_features(obs, nxt)
    assert features.shape == (1, 2 * PHYSICAL_OBS_DIM)
    assert torch.equal(features[0, :48], obs[0, :48])
    assert torch.equal(features[0, 48:], nxt[0, :48])


class _Data:
    root_link_lin_vel_b = torch.zeros(1, 3)
    root_link_ang_vel_b = torch.zeros(1, 3)


class _ContactData:
    found = torch.tensor([[True, True]])


class _BaseEnv:
    def __init__(self):
        self.num_envs = 1
        self.num_actions = 14
        self.device = torch.device("cpu")
        self.max_episode_length = 2
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.cfg = object()
        self.unwrapped = self
        self.scene = {
            "robot": type("Robot", (), {"data": _Data()})(),
            "feet_ground_contact": type("Contact", (), {"data": _ContactData()})(),
        }
        self.obs = TensorDict(
            {
                "actor": torch.zeros(1, 61),
                "critic": torch.zeros(1, 64),
            },
            batch_size=[1],
        )

    def get_observations(self):
        return self.obs.clone()

    def reset(self):
        return self.obs.clone(), {}

    def step(self, _actions):
        return self.obs.clone(), torch.zeros(1), torch.zeros(1, dtype=torch.long), {}

    def close(self):
        pass


def _write_reference(path):
    zeros = {
        "reference_joint_pos_rel": np.zeros((2, 14), np.float32),
        "reference_joint_vel": np.zeros((2, 14), np.float32),
        "reference_root_lin_vel_b": np.zeros((2, 3), np.float32),
        "reference_root_ang_vel_b": np.zeros((2, 3), np.float32),
        "reference_projected_gravity": np.zeros((2, 3), np.float32),
        "reference_foot_contact": np.ones((2, 2), np.float32),
        "reference_actions": np.zeros((2, 14), np.float32),
    }
    np.savez(path, **zeros)


@pytest.mark.parametrize("mode", ["deepmimic", "deepmimic_amp"])
def test_deepmimic_wrapper_injects_phase_and_rewards_exact_match(tmp_path, mode):
    path = tmp_path / "reference.npz"
    _write_reference(path)
    env = RunningReferenceWrapper(_BaseEnv(), str(path), mode=mode)
    observations = env.get_observations()
    assert observations["actor"][0, HEAD_COMMAND_START] == 0
    assert observations["actor"][0, HEAD_COMMAND_START + 1] == 1
    observations, rewards, dones, _ = env.step(torch.zeros(1, 14))
    assert torch.allclose(rewards, torch.ones(1))
    assert not dones.any()
    assert torch.allclose(
        observations["actor"][0, HEAD_COMMAND_START : HEAD_COMMAND_START + 2],
        torch.tensor([0.0, -1.0]),
        atol=1e-6,
    )
