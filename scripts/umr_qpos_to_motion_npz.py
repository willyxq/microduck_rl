"""Convert a UMR MicroDuck qpos npz into mjlab BeyondMimic motion.npz.

UMR writes MuJoCo qpos (xyz + wxyz + hinges) at the source clip fps. mjlab
tracking consumes per-control-step arrays:

    fps, joint_pos, joint_vel, body_pos_w, body_quat_w,
    body_lin_vel_w, body_ang_vel_w

This script resamples to the tracking control rate (50 Hz by default),
replays FK on the walk MJCF, and writes a local npz. No wandb upload.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import torch
import tyro
from tqdm import tqdm

from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML

DEFAULT_UMR_WALK = Path(
    "/home/william/Workspace/e1901/repro/UMR-run/output/"
    "microduck_retarget/humanoid_walk_character_microduck.npz"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "mjlab_microduck"
    / "motions"
    / "umr_humanoid_walk.npz"
)


def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    return a * (1 - blend) + b * blend


def _slerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(a)
    for i in range(a.shape[0]):
        out[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return out


def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def _interpolate_qpos(
    qpos: np.ndarray,
    input_fps: float,
    output_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample root pos / wxyz quat / hinges to ``output_fps``."""
    device = torch.device("cpu")
    motion = torch.from_numpy(np.asarray(qpos, dtype=np.float32)).to(device)
    if motion.ndim != 2 or motion.shape[1] < 8:
        raise ValueError(f"qpos must be (T, 7+n), got {tuple(motion.shape)}")

    base_pos = motion[:, :3]
    base_quat = motion[:, 3:7]
    dof_pos = motion[:, 7:]

    input_dt = 1.0 / float(input_fps)
    output_dt = 1.0 / float(output_fps)
    input_frames = int(motion.shape[0])
    duration = (input_frames - 1) * input_dt
    times = torch.arange(0, duration, output_dt, device=device, dtype=torch.float32)
    phase = times / duration
    index_0 = (phase * (input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(input_frames - 1))
    blend = phase * (input_frames - 1) - index_0.to(times.dtype)

    out_pos = _lerp(base_pos[index_0], base_pos[index_1], blend.unsqueeze(1))
    out_quat = _slerp(base_quat[index_0], base_quat[index_1], blend)
    out_dof = _lerp(dof_pos[index_0], dof_pos[index_1], blend.unsqueeze(1))
    return (
        out_pos.numpy(),
        out_quat.numpy(),
        out_dof.numpy(),
    )


def _fk_bodies(
    xml_path: Path,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    dof_pos: np.ndarray,
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    mj_joint_names = [model.joint(i).name for i in range(model.njnt)]
    missing = [name for name in joint_names if name not in mj_joint_names]
    if missing:
        raise KeyError(f"UMR joints missing from {xml_path}: {missing}")
    if dof_pos.shape[1] != len(joint_names):
        raise ValueError(
            f"dof width {dof_pos.shape[1]} != joint_names {len(joint_names)}"
        )

    qpos_adr = {name: int(model.joint(name).qposadr[0]) for name in joint_names}
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    n_frames = root_pos.shape[0]
    body_pos = np.zeros((n_frames, model.nbody - 1, 3), dtype=np.float32)
    body_quat = np.zeros((n_frames, model.nbody - 1, 4), dtype=np.float32)

    for t in tqdm(range(n_frames), desc="FK replay", unit="frame"):
        data.qpos[:] = 0.0
        data.qpos[0:3] = root_pos[t]
        data.qpos[3:7] = root_quat[t]
        for j, name in enumerate(joint_names):
            data.qpos[qpos_adr[name]] = dof_pos[t, j]
        mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos[1:]
        body_quat[t] = data.xquat[1:]

    return body_pos, body_quat, body_names


def convert(
    input_file: Path = DEFAULT_UMR_WALK,
    output_file: Path = DEFAULT_OUTPUT,
    output_fps: float = 50.0,
    robot_xml: Path | None = None,
) -> Path:
    src = np.load(input_file, allow_pickle=True)
    if "qpos" not in src.files:
        raise KeyError(f"{input_file} has no qpos; keys={src.files}")
    qpos = np.asarray(src["qpos"], dtype=np.float32)
    input_fps = float(np.asarray(src["fps"]).reshape(-1)[0]) if "fps" in src.files else 30.0
    joint_names = [str(n) for n in np.asarray(src["robot_joint_names"]).tolist()]
    xml_path = Path(robot_xml) if robot_xml is not None else MICRODUCK_WALK_XML

    root_pos, root_quat, dof_pos = _interpolate_qpos(qpos, input_fps, output_fps)
    output_dt = 1.0 / float(output_fps)
    joint_vel = np.gradient(dof_pos, output_dt, axis=0).astype(np.float32)

    body_pos, body_quat, body_names = _fk_bodies(
        xml_path, root_pos, root_quat, dof_pos, joint_names
    )
    body_lin_vel = np.gradient(body_pos, output_dt, axis=0).astype(np.float32)
    body_quat_t = torch.from_numpy(body_quat)
    body_ang_vel = torch.stack(
        [_so3_derivative(body_quat_t[:, b], output_dt) for b in range(body_quat.shape[1])],
        dim=1,
    ).numpy()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_file,
        fps=np.array([output_fps], dtype=np.float32),
        joint_pos=dof_pos.astype(np.float32),
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
        body_names=np.array(body_names),
        joint_names=np.array(joint_names),
        source_file=str(input_file),
        source_fps=np.array([input_fps], dtype=np.float32),
    )
    print(
        f"Wrote {output_file}  frames={dof_pos.shape[0]}  "
        f"{input_fps:g}→{output_fps:g} Hz  bodies={len(body_names)}  "
        f"joints={len(joint_names)}"
    )
    print(
        f"  root z [{root_pos[:, 2].min():.3f}, {root_pos[:, 2].max():.3f}]  "
        f"xy span {np.ptp(root_pos[:, :2], axis=0)}"
    )
    return output_file


def main(
    input_file: Path = DEFAULT_UMR_WALK,
    output_file: Path = DEFAULT_OUTPUT,
    output_fps: float = 50.0,
    robot_xml: Path | None = None,
) -> None:
    if not input_file.exists():
        raise FileNotFoundError(input_file)
    convert(input_file, output_file, output_fps, robot_xml)


if __name__ == "__main__":
    tyro.cli(main)
