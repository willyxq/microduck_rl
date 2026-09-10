#!/usr/bin/env python3
"""Collect a reusable motion reference from the Hannes running policy."""

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
    output_file: str
    speed: float = 2.2
    num_envs: int = 128
    warmup_s: float = 1.0
    duration_s: float = 4.0
    seed: int = 23
    task_id: str = "Mjlab-Running-Flat-MicroDuck"


def _yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main(cfg: Config) -> None:
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(cfg.task_id, play=True)
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.episode_length_s = cfg.warmup_s + cfg.duration_s + 1.0
    env_cfg.terminations.clear()
    command = env_cfg.commands["twist"]
    command.ranges.lin_vel_x = (cfg.speed, cfg.speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = 0.0
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
    obs = env.get_observations()
    initial_pos = robot.data.root_link_pos_w.clone()
    initial_yaw = _yaw(robot.data.root_link_quat_w).clone()
    warmup_steps = round(cfg.warmup_s / raw_env.step_dt)
    record_steps = round(cfg.duration_s / raw_env.step_dt)
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
    started = time.perf_counter()
    for step in range(warmup_steps + record_steps):
        current_obs = obs["actor"].clone()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if step < warmup_steps:
            continue
        records["obs"].append(current_obs)
        records["next_obs"].append(obs["actor"].clone())
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
    displacement = robot.data.root_link_pos_w - initial_pos
    heading = _yaw(robot.data.root_link_quat_w)
    heading_error = torch.atan2(
        torch.sin(heading - initial_yaw), torch.cos(heading - initial_yaw)
    ).abs()
    forward_speed = displacement[:, 0] / (cfg.warmup_s + cfg.duration_s)
    lateral_speed = displacement[:, 1].abs() / (cfg.warmup_s + cfg.duration_s)
    score = forward_speed - 0.4 * lateral_speed - 0.2 * heading_error
    selected = int(torch.argmax(score))

    output = Path(cfg.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "period_s": np.array(cfg.duration_s, dtype=np.float32),
        "step_dt": np.array(raw_env.step_dt, dtype=np.float32),
        "speed_command": np.array(cfg.speed, dtype=np.float32),
        "selected_env": np.array(selected, dtype=np.int64),
    }
    for key, value in stacked.items():
        arrays[f"reference_{key}"] = value[:, selected].cpu().numpy()
    # AMP uses diverse transitions from every collected rollout, not only the
    # canonical DeepMimic track.
    arrays["expert_obs"] = stacked["obs"].flatten(0, 1).cpu().numpy()
    arrays["expert_next_obs"] = stacked["next_obs"].flatten(0, 1).cpu().numpy()
    np.savez_compressed(output, **arrays)
    summary = {
        "checkpoint": cfg.checkpoint_file,
        "output": str(output),
        "device": device,
        "num_envs": cfg.num_envs,
        "frames": record_steps,
        "expert_transitions": cfg.num_envs * record_steps,
        "selected_env": selected,
        "selected_forward_speed_mps": float(forward_speed[selected]),
        "selected_lateral_speed_mps": float(lateral_speed[selected]),
        "selected_heading_error_deg": math.degrees(float(heading_error[selected])),
        "collection_seconds": time.perf_counter() - started,
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
