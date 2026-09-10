#!/usr/bin/env python3
"""Train a free-base MicroDuck runner with DeepMimic or AMP."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

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
    reference_file: str
    teacher_checkpoint: str
    output_dir: str
    iterations: int = 150
    num_envs: int = 1024
    speed: float = 2.2
    seed: int = 31
    task_id: str = "Mjlab-Running-Flat-MicroDuck"


def main(cfg: Config) -> None:
    if cfg.mode not in {"deepmimic", "amp"}:
        raise ValueError("mode must be deepmimic or amp")
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    reference = __import__("numpy").load(cfg.reference_file)
    reference_duration = float(reference["period_s"])

    env_cfg = load_env_cfg(cfg.task_id, play=False)
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.episode_length_s = reference_duration
    env_cfg.curriculum.clear()
    command = env_cfg.commands["twist"]
    command.ranges.lin_vel_x = (cfg.speed, cfg.speed)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.rel_standing_envs = 0.0
    command.rel_turn_in_place_envs = 0.0
    command.resampling_time_range = (reference_duration, reference_duration)

    agent_cfg = load_rl_cfg(cfg.task_id)
    train_cfg = asdict(agent_cfg)
    train_cfg["logger"] = "tensorboard"
    train_cfg["save_interval"] = 50
    train_cfg["experiment_name"] = f"running_{cfg.mode}"
    train_cfg["run_name"] = cfg.mode
    train_cfg["algorithm"]["symmetry_cfg"] = None
    if cfg.mode == "amp":
        train_cfg["algorithm"]["class_name"] = (
            "mjlab_microduck.imitation_rl:AmpPPO"
        )
        train_cfg["algorithm"].update(
            {
                "expert_data_path": cfg.reference_file,
                "amp_reward_weight": 0.7,
                "task_reward_weight": 0.3,
                "discriminator_learning_rate": 2e-4,
                "discriminator_updates": 4,
                "discriminator_batch_size": 4096,
            }
        )

    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    base_env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    env = RunningReferenceWrapper(base_env, cfg.reference_file, mode=cfg.mode)
    runner_cls = load_runner_cls(cfg.task_id) or OnPolicyRunner
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = runner_cls(env, train_cfg, log_dir=str(output_dir), device=device)
    # Warm start is explicit: it lets this compact experiment focus on the
    # imitation objectives instead of rediscovering standing and locomotion.
    runner.load(
        cfg.teacher_checkpoint,
        load_cfg={
            "actor": True,
            "critic": False,
            "optimizer": False,
            "iteration": False,
            "rnd": False,
        },
        strict=True,
        map_location=device,
    )

    started = time.perf_counter()
    runner.learn(num_learning_iterations=cfg.iterations)
    training_seconds = time.perf_counter() - started
    checkpoint_path = output_dir / "model_final.pt"
    runner.save(str(checkpoint_path))
    runner.export_policy_to_onnx(str(output_dir), f"{cfg.mode}_policy.onnx")

    policy = runner.get_inference_policy(device=device)
    # mjlab's actuator delay buffers are touched inside the runner's
    # inference-mode rollout, so reset them under the same mode.
    with torch.inference_mode():
        observations, _ = env.reset()
    steps = round(reference_duration / raw_env.step_dt)
    reward_sum = torch.zeros(cfg.num_envs, device=device)
    speed_sum = torch.zeros_like(reward_sum)
    fell = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
    for _ in range(steps):
        with torch.inference_mode():
            actions = policy(observations)
            observations, rewards, dones, _ = env.step(actions)
        reward_sum += rewards
        speed_sum += raw_env.scene["robot"].data.root_link_lin_vel_b[:, 0]
        fell |= dones.bool()

    metrics = {
        "mode": cfg.mode,
        "teacher_checkpoint": cfg.teacher_checkpoint,
        "reference_file": cfg.reference_file,
        "warm_started_from_teacher_actor": True,
        "device": device,
        "iterations": cfg.iterations,
        "num_envs": cfg.num_envs,
        "training_seconds": training_seconds,
        "mean_reward_per_step": float((reward_sum / steps).mean()),
        "mean_forward_speed_mps": float((speed_sum / steps).mean()),
        "survival_fraction": float((~fell).float().mean()),
        "checkpoint": str(checkpoint_path),
        "onnx": str(output_dir / f"{cfg.mode}_policy.onnx"),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    env.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
