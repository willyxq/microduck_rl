#!/usr/bin/env python3
"""Render a trained MicroDuck imitation policy to MP4."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort

from mjlab_microduck.imitation import (
    COMMAND_START,
    DEFAULT_POSE,
    OBS_DIM,
    load_motion,
)

SCENE = "src/mjlab_microduck/robot/microduck/scene.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def quat_rotate_inverse(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w = quat[0]
    xyz = quat[1:4]
    cross = np.cross(xyz, vector) * 2.0
    return vector - w * cross + np.cross(xyz, cross)


def main() -> None:
    args = parse_args()
    motion = load_motion(args.motion)
    model = mujoco.MjModel.from_xml_path(SCENE)
    model.opt.timestep = 0.005
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    data = mujoco.MjData(model)

    actuator_joint_ids = model.actuator_trnid[:, 0]
    joint_qpos = model.jnt_qposadr[actuator_joint_ids]
    joint_qvel = model.jnt_dofadr[actuator_joint_ids]
    home = DEFAULT_POSE.copy()
    free_joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
    )
    free_qpos = int(model.jnt_qposadr[free_joint])
    data.qpos[free_qpos : free_qpos + 7] = [0.0, 0.0, 0.125, 1.0, 0.0, 0.0, 0.0]
    data.qpos[joint_qpos] = home
    data.ctrl[:] = home
    mujoco.mj_forward(model, data)

    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    gyro_sensor = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel"
    )
    gyro_address = int(model.sensor_adr[gyro_sensor])
    session = ort.InferenceSession(
        str(args.policy), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    if session.get_inputs()[0].shape[-1] != OBS_DIM:
        raise ValueError(f"policy must accept {OBS_DIM} observations")

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.12]
    camera.distance = 0.62
    camera.azimuth = 140.0
    camera.elevation = -12.0

    last_action = np.zeros(model.nu, dtype=np.float32)
    frames: list[np.ndarray] = []
    control_dt = 0.02
    settle_s = 1.0
    next_frame_time = 0.0
    total_s = settle_s + args.duration
    root_heights: list[float] = []
    tilts: list[float] = []
    simulation_time = 0.0
    while simulation_time < total_s:
        phase = max(0.0, simulation_time - settle_s) / motion.period_s
        command = np.zeros(13, dtype=np.float32)
        command[0] = math.sin(2.0 * math.pi * phase)
        command[1] = math.cos(2.0 * math.pi * phase)
        projected_gravity = quat_rotate_inverse(
            data.xquat[trunk].astype(np.float32),
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )
        observation = np.concatenate(
            (
                data.sensordata[gyro_address : gyro_address + 3],
                projected_gravity,
                data.qpos[joint_qpos] - home,
                data.qvel[joint_qvel],
                last_action,
                command,
            )
        ).astype(np.float32)
        if simulation_time < settle_s:
            action = np.zeros(model.nu, dtype=np.float32)
        else:
            action = session.run(
                [output_name], {input_name: observation[None, :]}
            )[0][0].astype(np.float32)
        data.ctrl[:] = home + action
        last_action = action
        for _ in range(round(control_dt / model.opt.timestep)):
            mujoco.mj_step(model, data)
            simulation_time += model.opt.timestep
            if simulation_time + 1e-9 >= next_frame_time:
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render().copy())
                next_frame_time += 1.0 / args.fps
        root_heights.append(float(data.qpos[free_qpos + 2]))
        tilts.append(math.acos(float(np.clip(-projected_gravity[2], -1.0, 1.0))))

    renderer.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps, macro_block_size=2)
    stats = {
        "video": str(args.output),
        "duration_s": total_s,
        "frames": len(frames),
        "minimum_trunk_height_m": min(root_heights),
        "maximum_trunk_tilt_deg": math.degrees(max(tilts)),
        "final_xy_m": data.qpos[free_qpos : free_qpos + 2].tolist(),
        "finite_state": bool(np.all(np.isfinite(data.qpos))),
    }
    stats_path = args.output.with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
