#!/usr/bin/env python3
"""Create a dynamically feasible MicroDuck expert rollout from video offsets."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

import mjlab_microduck.tasks  # noqa: F401


@dataclass
class Config:
    checkpoint_file: str
    motion_file: str
    output_file: str
    num_envs: int = 512
    warmup_s: float = 1.0
    motion_scale: float = 1.0
    seed: int = 47
    task_id: str = "Mjlab-Running-Flat-MicroDuck"


def main(cfg: Config) -> None:
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    motion_data = np.load(cfg.motion_file)
    offsets = torch.as_tensor(
        motion_data["action_offsets"], dtype=torch.float32, device=device
    )
    period_s = float(motion_data["period_s"])

    env_cfg = load_env_cfg(cfg.task_id, play=True)
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.episode_length_s = cfg.warmup_s + period_s + 1.0
    env_cfg.terminations.clear()
    command = env_cfg.commands["twist"]
    command.ranges.lin_vel_x = (0.0, 0.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = 1.0
    command.resampling_time_range = (env_cfg.episode_length_s, env_cfg.episode_length_s)

    agent_cfg = load_rl_cfg(cfg.task_id)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(cfg.task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(cfg.checkpoint_file, map_location=device)
    policy = runner.get_inference_policy(device=device)

    robot = raw_env.scene["robot"]
    contact = raw_env.scene["feet_ground_contact"]
    observations = env.get_observations()
    warmup_steps = round(cfg.warmup_s / raw_env.step_dt)
    records: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "obs",
            "next_obs",
            "actions",
            "joint_pos_rel",
            "joint_vel",
            "root_lin_vel_b",
            "root_ang_vel_b",
            "projected_gravity",
            "foot_contact",
        )
    }
    max_tilt = torch.zeros(cfg.num_envs, device=device)
    min_height = torch.full((cfg.num_envs,), float("inf"), device=device)
    started = time.perf_counter()
    for step in range(warmup_steps + len(offsets)):
        current_obs = observations["actor"].clone()
        with torch.inference_mode():
            base_actions = policy(observations)
            if step >= warmup_steps:
                motion_step = step - warmup_steps
                actions = base_actions + cfg.motion_scale * offsets[motion_step]
            else:
                actions = base_actions
            observations, _, _, _ = env.step(actions)
        gravity_z = robot.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0)
        tilt = torch.acos((-gravity_z).clamp(-1.0, 1.0))
        max_tilt = torch.maximum(max_tilt, tilt)
        min_height = torch.minimum(min_height, robot.data.root_link_pos_w[:, 2])
        if step < warmup_steps:
            continue
        records["obs"].append(current_obs)
        records["next_obs"].append(observations["actor"].clone())
        records["actions"].append(actions.clone())
        records["joint_pos_rel"].append(
            robot.data.joint_pos - robot.data.default_joint_pos
        )
        records["joint_vel"].append(robot.data.joint_vel.clone())
        records["root_lin_vel_b"].append(robot.data.root_link_lin_vel_b.clone())
        records["root_ang_vel_b"].append(robot.data.root_link_ang_vel_b.clone())
        records["projected_gravity"].append(robot.data.projected_gravity_b.clone())
        records["foot_contact"].append(
            contact.data.found.reshape(cfg.num_envs, -1).float()
        )

    stacked = {key: torch.stack(value) for key, value in records.items()}
    survived = (max_tilt < math.radians(55.0)) & (min_height > 0.06)
    if not survived.any():
        raise RuntimeError(
            "no dynamically valid dance rollout; reduce --motion-scale"
        )
    score = min_height - 0.02 * max_tilt
    score = torch.where(survived, score, torch.full_like(score, -1e9))
    selected = int(torch.argmax(score))
    survivor_ids = torch.where(survived)[0]

    output = Path(cfg.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "period_s": np.array(period_s, np.float32),
        "step_dt": np.array(raw_env.step_dt, np.float32),
        "speed_command": np.array(0.0, np.float32),
        "selected_env": np.array(selected, np.int64),
        "source_action_offsets": offsets.cpu().numpy(),
    }
    for key, value in stacked.items():
        arrays[f"reference_{key}"] = value[:, selected].cpu().numpy()
    arrays["expert_obs"] = (
        stacked["obs"][:, survivor_ids].flatten(0, 1).cpu().numpy()
    )
    arrays["expert_next_obs"] = (
        stacked["next_obs"][:, survivor_ids].flatten(0, 1).cpu().numpy()
    )
    np.savez_compressed(output, **arrays)
    metrics = {
        "motion_file": cfg.motion_file,
        "checkpoint": cfg.checkpoint_file,
        "output": str(output),
        "num_envs": cfg.num_envs,
        "surviving_envs": int(survived.sum()),
        "survival_fraction": float(survived.float().mean()),
        "expert_transitions": int(len(offsets) * len(survivor_ids)),
        "selected_env": selected,
        "selected_max_tilt_deg": math.degrees(float(max_tilt[selected])),
        "selected_minimum_height_m": float(min_height[selected]),
        "period_s": period_s,
        "motion_scale": cfg.motion_scale,
        "collection_seconds": time.perf_counter() - started,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
