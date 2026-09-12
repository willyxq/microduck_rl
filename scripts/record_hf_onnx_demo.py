#!/usr/bin/env python3
"""Headless ONNX replay for any 61-D MicroDuck policy.

Loads one ONNX through PolicyInference (new 13-D command, projected gravity)
and writes a short MP4. Used to film Hugging Face policies locally.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_policy import (  # noqa: E402
    MICRODUCK_BALL_XML,
    MICRODUCK_ROLLERS_XML,
    MICRODUCK_XML,
    PolicyInference,
)


def _parse_timeline(raw: str | None) -> list[tuple[float, float, float, float]]:
    if not raw:
        return []
    events: list[tuple[float, float, float, float]] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        t_s, cmd = part.split(":", 1)
        nums = [float(x) for x in cmd.split(",")]
        while len(nums) < 3:
            nums.append(0.0)
        events.append((float(t_s), nums[0], nums[1], nums[2]))
    events.sort()
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Record one HF ONNX policy to MP4")
    parser.add_argument("--onnx", required=True, help="Policy ONNX")
    parser.add_argument("--output", required=True, help="Output MP4")
    parser.add_argument(
        "--kind",
        default="walking",
        choices=(
            "walking",
            "standing",
            "sitstand",
            "ground_pick",
            "roulade",
            "kick_left",
            "kick_right",
        ),
    )
    parser.add_argument("--scene", default="flat", choices=("flat", "rollers", "ball"))
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument(
        "--timeline",
        default=None,
        help='Command changes as "t:vx,vy,wz;t:..." (seconds from start)',
    )
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--stand-onnx", default=None, help="Optional standing ONNX")
    parser.add_argument("--trigger-at", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)

    if args.scene == "rollers":
        xml_path = MICRODUCK_ROLLERS_XML
    elif args.scene == "ball" or args.kind in ("kick_left", "kick_right"):
        xml_path = MICRODUCK_BALL_XML
    else:
        xml_path = MICRODUCK_XML

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    from bam.model import load_model

    kt = load_model(motor_name="xl330", model="m6").kt.value
    torque_limit = kt * 1.75
    model.actuator_forcerange[:, 0] = -torque_limit
    model.actuator_forcerange[:, 1] = torque_limit
    model.actuator_forcelimited[:] = 1

    kwargs: dict = {
        "model": model,
        "data": data,
        "action_scale": args.action_scale,
        "new_cmd_obs": True,
        "use_projected_gravity": True,
    }
    if args.kind == "standing":
        kwargs["standing_onnx_path"] = args.onnx
    elif args.kind == "sitstand":
        kwargs["sitstand_onnx_path"] = args.onnx
    elif args.kind == "ground_pick":
        kwargs["standing_onnx_path"] = args.stand_onnx or args.onnx
        kwargs["ground_pick_onnx_path"] = args.onnx
    elif args.kind == "roulade":
        kwargs["standing_onnx_path"] = args.stand_onnx or args.onnx
        kwargs["roulade_onnx_path"] = args.onnx
    elif args.kind == "kick_left":
        kwargs["standing_onnx_path"] = args.stand_onnx or args.onnx
        kwargs["kick_left_onnx_path"] = args.onnx
    elif args.kind == "kick_right":
        kwargs["standing_onnx_path"] = args.stand_onnx or args.onnx
        kwargs["kick_right_onnx_path"] = args.onnx
    else:
        kwargs["walking_onnx_path"] = args.onnx
        if args.stand_onnx:
            kwargs["standing_onnx_path"] = args.stand_onnx

    policy = PolicyInference(**kwargs)
    policy.set_vel_cmd(args.vx, args.vy, args.wz)

    if args.scene == "rollers":
        import re

        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name and re.match(r"^passive_.*", name):
                model.dof_frictionloss[model.jnt_dofadr[j]] = 0.003

    freejoint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qpos_adr = model.jnt_qposadr[freejoint_id]
    data.qpos[qpos_adr + 2] = 0.1385 if args.scene == "rollers" else 0.125
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1, 0, 0, 0]
    for i, qpos_idx in enumerate(policy.joint_qpos_indices):
        data.qpos[qpos_idx] = policy.default_pose[i]
    data.ctrl[:] = policy.default_pose
    mujoco.mj_forward(model, data)

    decimation = 4
    control_dt = decimation * model.opt.timestep
    total_steps = int(args.seconds / control_dt)
    frame_every = max(1, int(round((1.0 / args.fps) / control_dt)))
    timeline = _parse_timeline(args.timeline)
    timeline_i = 0
    triggered = False

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 1.8
    camera.elevation = -15
    camera.azimuth = 120
    camera.lookat[:] = [0.0, 0.0, 0.12]

    frames: list[np.ndarray] = []
    print(f"Recording {args.kind} {args.onnx} -> {out} ({args.seconds:.1f}s)")

    for step in range(total_steps):
        t = step * control_dt
        while timeline_i < len(timeline) and t + 1e-9 >= timeline[timeline_i][0]:
            _, vx, vy, wz = timeline[timeline_i]
            policy.set_vel_cmd(vx, vy, wz)
            timeline_i += 1

        if not triggered and t >= args.trigger_at:
            if args.kind == "sitstand":
                policy.toggle_sit()
            elif args.kind == "ground_pick":
                policy.trigger_ground_pick()
            elif args.kind == "roulade":
                policy.trigger_behavior("roulade")
            elif args.kind == "kick_left":
                policy.trigger_behavior("kick_left")
            elif args.kind == "kick_right":
                policy.trigger_behavior("kick_right")
            triggered = True
        if args.kind == "sitstand" and triggered and t >= args.trigger_at + 3.0 and policy.sit_mode:
            policy.toggle_sit()

        if args.kind == "ground_pick":
            policy.update_ground_pick_phase(control_dt)
        if policy.behavior_mode is not None:
            policy.behavior_time_left -= control_dt
            if policy.behavior_time_left <= 0:
                policy._end_behavior()

        action = policy.infer()
        policy.apply_action(action)
        for _ in range(decimation):
            mujoco.mj_step(model, data)

        if step % frame_every == 0:
            camera.lookat[:] = data.xpos[policy.trunk_base_id]
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

    iio.imwrite(out, frames, fps=args.fps, codec="libx264", quality=7)
    print(f"Saved {len(frames)} frames -> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
