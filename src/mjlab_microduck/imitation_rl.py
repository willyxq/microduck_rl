"""DeepMimic reference rewards and AMP-PPO for MicroDuck."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from rsl_rl.algorithms import PPO
from torch import nn
from torch.nn import functional as F

PHYSICAL_OBS_DIM = 48
HEAD_COMMAND_START = 51


class RunningReferenceWrapper:
    """Replace task rewards with DeepMimic tracking rewards.

    The wrapped simulator remains the regular free-base MicroDuck environment.
    Only the reward and two otherwise-unused head-command observation slots are
    changed.
    """

    def __init__(self, env, reference_file: str, mode: str = "deepmimic"):
        if mode not in {"deepmimic", "amp", "deepmimic_amp"}:
            raise ValueError(f"unknown imitation mode: {mode}")
        self.env = env
        self.mode = mode
        data = np.load(reference_file)
        self.reference_length = len(data["reference_joint_pos_rel"])
        self.reference_joint_pos = self._tensor(data["reference_joint_pos_rel"])
        self.reference_joint_vel = self._tensor(data["reference_joint_vel"])
        self.reference_root_lin_vel = self._tensor(data["reference_root_lin_vel_b"])
        self.reference_root_ang_vel = self._tensor(data["reference_root_ang_vel_b"])
        self.reference_gravity = self._tensor(data["reference_projected_gravity"])
        self.reference_contact = self._tensor(data["reference_foot_contact"])
        self.reference_actions = self._tensor(data["reference_actions"])
        self.reference_step = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _inject_phase(self, observations):
        if self.mode not in {"deepmimic", "deepmimic_amp"}:
            return observations
        observations = observations.clone()
        phase = self.reference_step.float() / self.reference_length
        observations["actor"][:, HEAD_COMMAND_START] = torch.sin(
            2.0 * torch.pi * phase
        )
        observations["actor"][:, HEAD_COMMAND_START + 1] = torch.cos(
            2.0 * torch.pi * phase
        )
        return observations

    def get_observations(self):
        return self._inject_phase(self.env.get_observations())

    def reset(self):
        observations, extras = self.env.reset()
        self.reference_step.zero_()
        return self._inject_phase(observations), extras

    def step(self, actions):
        observations, task_rewards, dones, extras = self.env.step(actions)
        if self.mode in {"deepmimic", "deepmimic_amp"}:
            idx = self.reference_step
            actor_obs = observations["actor"]
            pose_error = torch.mean(
                (actor_obs[:, 6:20] - self.reference_joint_pos[idx]) ** 2, dim=1
            )
            velocity_error = torch.mean(
                (actor_obs[:, 20:34] - self.reference_joint_vel[idx]) ** 2, dim=1
            )
            gravity_error = torch.mean(
                (actor_obs[:, 3:6] - self.reference_gravity[idx]) ** 2, dim=1
            )
            robot = self.unwrapped.scene["robot"]
            root_velocity_error = torch.mean(
                (robot.data.root_link_lin_vel_b - self.reference_root_lin_vel[idx])
                ** 2,
                dim=1,
            )
            root_angular_error = torch.mean(
                (robot.data.root_link_ang_vel_b - self.reference_root_ang_vel[idx])
                ** 2,
                dim=1,
            )
            contact = self.unwrapped.scene["feet_ground_contact"].data.found.reshape(
                self.num_envs, -1
            )
            contact_match = (
                contact == (self.reference_contact[idx] > 0.5)
            ).float().mean(dim=1)
            action_error = torch.mean(
                (actions - self.reference_actions[idx]) ** 2, dim=1
            )
            rewards = (
                0.40 * torch.exp(-35.0 * pose_error)
                + 0.10 * torch.exp(-0.08 * velocity_error)
                + 0.20 * torch.exp(-2.0 * root_velocity_error)
                + 0.08 * torch.exp(-0.2 * root_angular_error)
                + 0.08 * torch.exp(-12.0 * gravity_error)
                + 0.08 * contact_match
                + 0.06 * torch.exp(-20.0 * action_error)
            )
            rewards = torch.where(dones.bool(), torch.zeros_like(rewards), rewards)
            extras.setdefault("log", {})["deepmimic_reward"] = rewards.mean()
        else:
            rewards = task_rewards

        self.reference_step = (self.reference_step + 1) % self.reference_length
        self.reference_step = torch.where(
            dones.bool(), torch.zeros_like(self.reference_step), self.reference_step
        )
        return self._inject_phase(observations), rewards, dones, extras

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.env.cfg

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def num_actions(self):
        return self.env.num_actions

    @property
    def device(self):
        return self.env.device

    @property
    def max_episode_length(self):
        return self.env.max_episode_length

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def close(self):
        self.env.close()


def transition_features(
    observations: torch.Tensor, next_observations: torch.Tensor
) -> torch.Tensor:
    """AMP state-transition features, excluding commands."""
    return torch.cat(
        (
            observations[..., :PHYSICAL_OBS_DIM],
            next_observations[..., :PHYSICAL_OBS_DIM],
        ),
        dim=-1,
    )


class AmpPPO(PPO):
    """PPO augmented with a standard adversarial motion-prior reward."""

    def __init__(
        self,
        *args,
        expert_data_path: str,
        amp_reward_weight: float = 0.7,
        task_reward_weight: float = 0.3,
        discriminator_learning_rate: float = 2e-4,
        discriminator_updates: int = 4,
        discriminator_batch_size: int = 4096,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        data = np.load(Path(expert_data_path))
        expert = transition_features(
            torch.as_tensor(data["expert_obs"], dtype=torch.float32),
            torch.as_tensor(data["expert_next_obs"], dtype=torch.float32),
        )
        self.expert_features = expert.to(self.device)
        self.feature_mean = self.expert_features.mean(0)
        self.feature_std = self.expert_features.std(0).clamp_min(1e-4)
        feature_dim = self.expert_features.shape[1]
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        ).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=discriminator_learning_rate
        )
        self.amp_reward_weight = amp_reward_weight
        self.task_reward_weight = task_reward_weight
        self.discriminator_updates = discriminator_updates
        self.discriminator_batch_size = discriminator_batch_size
        self.policy_feature_batches: list[torch.Tensor] = []
        self.latest_amp_reward = 0.0

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.feature_mean) / self.feature_std

    def process_env_step(self, obs, rewards, dones, extras):
        features = transition_features(
            self.transition.observations["actor"], obs["actor"]
        )
        with torch.no_grad():
            logits = self.discriminator(self._normalize_features(features)).squeeze(-1)
            # -log(1-D), where D is the probability of an expert transition.
            amp_reward = F.softplus(logits).clamp(max=10.0)
        self.latest_amp_reward = float(amp_reward.mean())
        self.policy_feature_batches.append(features.detach())
        combined_rewards = (
            self.task_reward_weight * rewards
            + self.amp_reward_weight * amp_reward
        )
        extras.setdefault("log", {})["amp_reward"] = amp_reward.mean()
        super().process_env_step(obs, combined_rewards, dones, extras)

    def _update_discriminator(self) -> tuple[float, float]:
        policy = torch.cat(self.policy_feature_batches, dim=0)
        mean_loss = 0.0
        mean_accuracy = 0.0
        for _ in range(self.discriminator_updates):
            count = min(
                self.discriminator_batch_size,
                len(policy),
                len(self.expert_features),
            )
            policy_idx = torch.randint(len(policy), (count,), device=self.device)
            expert_idx = torch.randint(
                len(self.expert_features), (count,), device=self.device
            )
            policy_batch = self._normalize_features(policy[policy_idx])
            expert_batch = self._normalize_features(
                self.expert_features[expert_idx]
            )
            policy_logits = self.discriminator(policy_batch)
            expert_logits = self.discriminator(expert_batch)
            loss = F.binary_cross_entropy_with_logits(
                expert_logits, torch.ones_like(expert_logits)
            ) + F.binary_cross_entropy_with_logits(
                policy_logits, torch.zeros_like(policy_logits)
            )
            self.discriminator_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 5.0)
            self.discriminator_optimizer.step()
            with torch.no_grad():
                accuracy = 0.5 * (
                    (expert_logits > 0).float().mean()
                    + (policy_logits < 0).float().mean()
                )
            mean_loss += float(loss)
            mean_accuracy += float(accuracy)
        self.policy_feature_batches.clear()
        return (
            mean_loss / self.discriminator_updates,
            mean_accuracy / self.discriminator_updates,
        )

    def update(self):
        discriminator_loss, discriminator_accuracy = self._update_discriminator()
        losses = super().update()
        losses["discriminator"] = discriminator_loss
        losses["discriminator_accuracy"] = discriminator_accuracy
        losses["amp_reward"] = self.latest_amp_reward
        return losses

    def save(self):
        state = super().save()
        state["amp_discriminator_state_dict"] = self.discriminator.state_dict()
        state["amp_discriminator_optimizer_state_dict"] = (
            self.discriminator_optimizer.state_dict()
        )
        return state

    def load(self, loaded_dict, load_cfg, strict):
        iteration = super().load(loaded_dict, load_cfg, strict)
        if "amp_discriminator_state_dict" in loaded_dict:
            self.discriminator.load_state_dict(
                loaded_dict["amp_discriminator_state_dict"]
            )
            self.discriminator_optimizer.load_state_dict(
                loaded_dict["amp_discriminator_optimizer_state_dict"]
            )
        return iteration
