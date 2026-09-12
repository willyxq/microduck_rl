#!/usr/bin/env python3
"""Replay reconstructed MicroDuck poses on a virtual gantry.

This is the visual teacher for filmed-duck retargeting: joints and root yaw
are written directly into qpos. A standing VelStand overlay cannot show the
90° whole-body turn in the Korean duck-dance clip.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import tyro

from mjlab_microduck.imitation import DEFAULT_POSE

SCENE = "src/mjlab_microduck/robot/microduck/scene.xml"


@dataclass
class Config:
    motion_file: str
    output_file: str
    width: int = 480
    height: int = 720
    fps: int = 30
    scene_file: str = SCENE
    azimuth: float = 180.0
    elevation: float = -8.0
    distance: float = 0.55


def _yaw_quaternion(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def main(cfg: Config) -> None:
    motion = np.load(cfg.motion_file)
    joints = motion["joint_pos"] if "joint_pos" in motion.files else (
        DEFAULT_POSE + motion["action_offsets"]
    )
    root_yaw = (
        motion["root_yaw"]
        if "root_yaw" in motion.files
        else np.zeros(len(joints), dtype=np.float32)
    )
    time = motion["target_time_s"]
    model = mujoco.MjModel.from_xml_path(cfg.scene_file)
    model.vis.global_.offwidth = cfg.width
    model.vis.global_.offheight = cfg.height
    data = mujoco.MjData(model)
    joint_qpos = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    free_qpos = int(model.jnt_qposadr[free])

    renderer = mujoco.Renderer(model, height=cfg.height, width=cfg.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.12]
    camera.distance = cfg.distance
    camera.azimuth = cfg.azimuth
    camera.elevation = cfg.elevation

    frames: list[np.ndarray] = []
    duration = float(time[-1] + (time[1] - time[0]))
    next_frame = 0.0
    while next_frame < duration - 1e-9:
        index = int(np.clip(np.searchsorted(time, next_frame), 0, len(joints) - 1))
        data.qpos[free_qpos : free_qpos + 3] = [0.0, 0.0, 0.12]
        data.qpos[free_qpos + 3 : free_qpos + 7] = _yaw_quaternion(float(root_yaw[index]))
        data.qpos[joint_qpos] = joints[index]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())
        next_frame += 1.0 / cfg.fps

    output = Path(cfg.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, frames, fps=cfg.fps, macro_block_size=2)
    metrics = {
        "video": str(output),
        "frames": len(frames),
        "duration_s": duration,
        "head_yaw_range_rad": float(np.ptp(joints[:, 7])),
        "root_yaw_range_rad": float(np.ptp(root_yaw)),
        "maximum_joint_rad": float(np.max(np.abs(joints))),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    renderer.close()


if __name__ == "__main__":
    main(tyro.cli(Config))
