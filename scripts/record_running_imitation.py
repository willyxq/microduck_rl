#!/usr/bin/env python3
"""Record teacher, DeepMimic, or AMP running ONNX policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_policy import MICRODUCK_XML, PolicyInference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode", required=True, choices=("teacher", "deepmimic", "amp")
    )
    parser.add_argument("--speed", type=float, default=2.2)
    parser.add_argument("--reference-period", type=float, default=4.0)
    parser.add_argument("--stand-seconds", type=float, default=1.0)
    parser.add_argument("--run-seconds", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(MICRODUCK_XML)
    model.opt.timestep = 0.005
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)
    policy = PolicyInference(
        model,
        data,
        walking_onnx_path=args.policy,
        new_cmd_obs=True,
        use_projected_gravity=True,
    )

    freejoint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    root_qpos = int(model.jnt_qposadr[freejoint])
    root_qvel = int(model.jnt_dofadr[freejoint])
    data.qpos[root_qpos : root_qpos + 7] = [0, 0, 0.125, 1, 0, 0, 0]
    data.qpos[policy.joint_qpos_indices] = policy.default_pose
    data.ctrl[:] = policy.default_pose
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 0.72
    camera.elevation = -14
    camera.azimuth = 130

    control_dt = 0.02
    total_steps = round((args.stand_seconds + args.run_seconds) / control_dt)
    stand_steps = round(args.stand_seconds / control_dt)
    frames = []
    next_frame_time = 0.0
    speed_samples = []
    max_tilt = 0.0
    min_height = float("inf")
    for step in range(total_steps):
        run_time = max(0.0, step * control_dt - args.stand_seconds)
        command = np.zeros(13, dtype=np.float32)
        if step >= stand_steps:
            command[0] = args.speed
            if args.mode == "deepmimic":
                phase = run_time / args.reference_period
                command[3] = math.sin(2 * math.pi * phase)
                command[4] = math.cos(2 * math.pi * phase)
        policy.command = command
        action = policy.infer()
        policy.apply_action(action)
        for _ in range(4):
            mujoco.mj_step(model, data)
        if step * control_dt + 1e-9 >= next_frame_time:
            camera.lookat[:] = data.xpos[policy.trunk_base_id]
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
            next_frame_time += 1.0 / args.fps
        gravity = policy.get_projected_gravity()
        max_tilt = max(
            max_tilt,
            math.degrees(math.acos(float(np.clip(-gravity[2], -1.0, 1.0)))),
        )
        min_height = min(min_height, float(data.qpos[root_qpos + 2]))
        if step >= stand_steps:
            quat = data.qpos[root_qpos + 3 : root_qpos + 7].astype(np.float32)
            world_velocity = data.qvel[root_qvel : root_qvel + 3].astype(np.float32)
            speed_samples.append(float(policy.quat_rotate_inverse(quat, world_velocity)[0]))

    renderer.close()
    iio.imwrite(output, frames, fps=args.fps, codec="libx264", quality=8)
    metrics = {
        "mode": args.mode,
        "policy": args.policy,
        "video": str(output),
        "mean_body_forward_speed_mps": float(np.mean(speed_samples)),
        "minimum_trunk_height_m": min_height,
        "maximum_tilt_deg": max_tilt,
        "finite_state": bool(np.all(np.isfinite(data.qpos))),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
