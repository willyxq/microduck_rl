#!/usr/bin/env python3
"""Record a running checkpoint in the same mjlab/BAM physics used to train it."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.runners import OnPolicyRunner

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.imitation_rl import RunningReferenceWrapper


@dataclass
class Config:
    mode: str
    checkpoint_file: str
    reference_file: str
    output_file: str
    speed: float = 2.2
    duration_s: float = 8.0
    seed: int = 0
    fps: int = 30
    task_id: str = "Mjlab-Running-Flat-MicroDuck"


def main(cfg: Config) -> None:
    if cfg.mode not in {"teacher", "deepmimic", "amp"}:
        raise ValueError("mode must be teacher, deepmimic, or amp")
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = load_env_cfg(cfg.task_id, play=True)
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = cfg.duration_s + 1.0
    env_cfg.terminations.clear()
    env_cfg.viewer.distance = 0.62
    env_cfg.viewer.azimuth = 135.0
    env_cfg.viewer.elevation = -14.0
    command = env_cfg.commands["twist"]
    command.ranges.lin_vel_x = (cfg.speed, cfg.speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (cfg.duration_s + 1.0, cfg.duration_s + 1.0)

    agent_cfg = load_rl_cfg(cfg.task_id)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    base_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    wrapper_mode = "deepmimic" if cfg.mode == "deepmimic" else "amp"
    env = RunningReferenceWrapper(base_env, cfg.reference_file, mode=wrapper_mode)
    runner_cls = load_runner_cls(cfg.task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        cfg.checkpoint_file,
        load_cfg={
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        },
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    observations = env.get_observations()
    robot = raw_env.scene["robot"]
    frames: list[np.ndarray] = []
    next_frame_s = 0.0
    steps = round(cfg.duration_s / raw_env.step_dt)
    warmup_steps = round(1.0 / raw_env.step_dt)
    speeds = []
    max_tilt = 0.0
    min_height = float("inf")
    for step in range(steps):
        with torch.inference_mode():
            actions = policy(observations)
            observations, _, _, _ = env.step(actions)
        elapsed = (step + 1) * raw_env.step_dt
        if elapsed + 1e-9 >= next_frame_s:
            frame = raw_env.render()
            frames.append(np.asarray(frame).copy())
            next_frame_s += 1.0 / cfg.fps
        gravity_z = float(robot.data.projected_gravity_b[0, 2])
        max_tilt = max(
            max_tilt,
            math.degrees(math.acos(float(np.clip(-gravity_z, -1.0, 1.0)))),
        )
        min_height = min(min_height, float(robot.data.root_link_pos_w[0, 2]))
        if step >= warmup_steps:
            speeds.append(float(robot.data.root_link_lin_vel_b[0, 0]))

    output = Path(cfg.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, frames, fps=cfg.fps, codec="libx264", quality=8)
    metrics = {
        "mode": cfg.mode,
        "checkpoint": cfg.checkpoint_file,
        "physics": "mjlab_mujoco_warp_bam",
        "video": str(output),
        "mean_body_forward_speed_mps": float(np.mean(speeds)),
        "minimum_trunk_height_m": min_height,
        "maximum_tilt_deg": max_tilt,
        "finite_state": bool(torch.isfinite(robot.data.root_link_pos_w).all()),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
