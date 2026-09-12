#!/usr/bin/env python3
"""Retarget a human duck-step clip onto MicroDuck hip and knee offsets.

MicroDuck has no independent waist. The trunk is a rigid free body; hip yaw
and hip roll are the only joints that can approximate a waddle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from scipy.ndimage import gaussian_filter1d


@dataclass
class Config:
    video_file: str
    output_file: str
    start_s: float = 8.0
    duration_s: float = 8.0
    target_hz: float = 50.0
    model: str = "yolo11n-pose.pt"
    device: str = "0"


def _fill_missing(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = values.copy()
    x = np.arange(len(values))
    for joint in range(values.shape[1]):
        for axis in range(values.shape[2]):
            mask = valid[:, joint] & np.isfinite(values[:, joint, axis])
            if mask.sum() < 2:
                raise RuntimeError(f"insufficient detections for keypoint {joint}")
            result[:, joint, axis] = np.interp(
                x, x[mask], values[mask, joint, axis]
            )
    return result


def _center_scale(feature: np.ndarray, scale: float) -> np.ndarray:
    return (feature - np.median(feature)) / max(scale, 1e-6)


def _knee_flexion(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> np.ndarray:
    upper = hip - knee
    lower = ankle - knee
    cosine = np.sum(upper * lower, axis=1) / np.maximum(
        np.linalg.norm(upper, axis=1) * np.linalg.norm(lower, axis=1), 1e-6
    )
    return np.pi - np.arccos(np.clip(cosine, -1.0, 1.0))


def retarget_duck_step(
    keypoints: np.ndarray,
    source_fps: float,
    target_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map COCO keypoints to 14 MicroDuck action offsets for a duck walk."""
    shoulder_mid = 0.5 * (keypoints[:, 5] + keypoints[:, 6])
    hip_mid = 0.5 * (keypoints[:, 11] + keypoints[:, 12])
    ankle_mid = 0.5 * (keypoints[:, 15] + keypoints[:, 16])
    torso = np.linalg.norm(shoulder_mid - hip_mid, axis=1).clip(20.0)
    hip_width = np.linalg.norm(keypoints[:, 11] - keypoints[:, 12], axis=1).clip(12.0)
    scale = float(np.median(torso))

    sway = np.clip(_center_scale(hip_mid[:, 0] - ankle_mid[:, 0], scale), -1.0, 1.0)
    hip_line = keypoints[:, 12] - keypoints[:, 11]
    hip_twist = np.clip(
        np.arctan2(hip_line[:, 1], hip_line[:, 0])
        - np.median(np.arctan2(hip_line[:, 1], hip_line[:, 0])),
        -0.6,
        0.6,
    )
    foot_spread = np.clip(
        _center_scale(
            np.linalg.norm(keypoints[:, 15] - keypoints[:, 16], axis=1) / hip_width,
            0.35,
        ),
        -1.0,
        1.0,
    )
    leg_height = (ankle_mid[:, 1] - hip_mid[:, 1]) / torso
    low_leg, high_leg = np.quantile(leg_height, [0.1, 0.9])
    squat = np.clip(
        (high_leg - leg_height) / max(high_leg - low_leg, 0.08),
        0.0,
        1.2,
    )
    left_flex = _knee_flexion(keypoints[:, 11], keypoints[:, 13], keypoints[:, 15])
    right_flex = _knee_flexion(keypoints[:, 12], keypoints[:, 14], keypoints[:, 16])
    left_bend = np.clip(_center_scale(left_flex, 0.35), -1.0, 1.0)
    right_bend = np.clip(_center_scale(right_flex, 0.35), -1.0, 1.0)
    left_step = np.clip(
        _center_scale(keypoints[:, 15, 0] - keypoints[:, 11, 0], scale), -0.8, 0.8
    )
    right_step = np.clip(
        _center_scale(keypoints[:, 16, 0] - keypoints[:, 12, 0], scale), -0.8, 0.8
    )
    left_lift = np.clip(
        (np.quantile(keypoints[:, 15, 1], 0.9) - keypoints[:, 15, 1]) / (0.18 * scale),
        0.0,
        1.0,
    )
    right_lift = np.clip(
        (np.quantile(keypoints[:, 16, 1], 0.9) - keypoints[:, 16, 1]) / (0.18 * scale),
        0.0,
        1.0,
    )
    head_side = np.clip(
        _center_scale(keypoints[:, 0, 0] - shoulder_mid[:, 0], hip_width[0]),
        -0.8,
        0.8,
    )

    source_time = np.arange(len(keypoints), dtype=np.float32) / source_fps
    target_time = np.arange(
        0.0, len(keypoints) / source_fps, 1.0 / target_hz, dtype=np.float32
    )

    def sample(feature: np.ndarray) -> np.ndarray:
        return np.interp(target_time, source_time, feature).astype(np.float32)

    sway_t = sample(sway)
    twist_t = sample(hip_twist)
    spread_t = sample(foot_spread)
    squat_t = sample(squat)
    left_bend_t = sample(left_bend)
    right_bend_t = sample(right_bend)
    left_step_t = sample(left_step)
    right_step_t = sample(right_step)
    left_lift_t = sample(left_lift)
    right_lift_t = sample(right_lift)
    head_side_t = sample(head_side)

    actions = np.zeros((len(target_time), 14), dtype=np.float32)
    # Opposite hip yaw approximates the missing waist twist / duck-toed waddle.
    actions[:, 0] = 0.28 * sway_t + 0.20 * twist_t + 0.12 * left_step_t
    actions[:, 9] = -0.28 * sway_t - 0.20 * twist_t - 0.12 * right_step_t
    # Same-sign hip roll shifts weight; this is the closest "waist lean".
    actions[:, 1] = 0.28 * sway_t + 0.08 * spread_t
    actions[:, 10] = 0.28 * sway_t - 0.08 * spread_t
    # Pitch/knee/ankle carry the squat and low duck-step.
    actions[:, 2] = -0.16 * squat_t + 0.18 * left_lift_t
    actions[:, 3] = -0.26 * squat_t - 0.16 * left_lift_t - 0.12 * left_bend_t
    actions[:, 4] = 0.14 * squat_t - 0.08 * left_lift_t
    actions[:, 11] = 0.16 * squat_t - 0.18 * right_lift_t
    actions[:, 12] = 0.26 * squat_t + 0.16 * right_lift_t + 0.12 * right_bend_t
    actions[:, 13] = -0.14 * squat_t + 0.08 * right_lift_t
    actions[:, 7] = 0.22 * sway_t + 0.10 * head_side_t
    actions[:, 8] = 0.16 * sway_t + 0.12 * twist_t
    actions = gaussian_filter1d(actions, sigma=1.0, axis=0, mode="nearest")

    blend = min(round(0.35 * target_hz), len(actions) // 4)
    if blend > 1:
        midpoint = 0.5 * (actions[0] + actions[-1])
        ramp = np.linspace(0.0, 1.0, blend, dtype=np.float32)[:, None]
        actions[:blend] = midpoint * (1 - ramp) + actions[:blend] * ramp
        actions[-blend:] = actions[-blend:] * (1 - ramp) + midpoint * ramp
    return target_time, actions


def main(cfg: Config) -> None:
    import cv2
    from ultralytics import YOLO

    capture = cv2.VideoCapture(cfg.video_file)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.set(cv2.CAP_PROP_POS_MSEC, cfg.start_s * 1000.0)
    frame_count = round(cfg.duration_s * fps)
    frames = []
    for _ in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < 2:
        raise RuntimeError("duck-step segment contains fewer than two frames")

    model = YOLO(cfg.model)
    predictions = model.predict(
        frames, device=cfg.device, verbose=False, classes=[0], batch=16
    )
    keypoints = np.full((len(frames), 17, 2), np.nan, dtype=np.float32)
    confidence = np.zeros((len(frames), 17), dtype=np.float32)
    for frame_index, prediction in enumerate(predictions):
        if prediction.boxes is None or len(prediction.boxes) == 0:
            continue
        scores = prediction.boxes.conf.detach().cpu().numpy()
        person = int(np.argmax(scores))
        keypoints[frame_index] = (
            prediction.keypoints.xy[person].detach().cpu().numpy()
        )
        if prediction.keypoints.conf is not None:
            confidence[frame_index] = (
                prediction.keypoints.conf[person].detach().cpu().numpy()
            )
    valid = confidence > 0.35
    keypoints = _fill_missing(keypoints, valid)
    keypoints = gaussian_filter1d(keypoints, sigma=1.0, axis=0, mode="nearest")
    target_time, actions = retarget_duck_step(keypoints, fps, cfg.target_hz)

    output = Path(cfg.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        action_offsets=actions,
        target_time_s=target_time,
        source_keypoints=keypoints,
        source_confidence=confidence,
        source_fps=np.array(fps, np.float32),
        target_hz=np.array(cfg.target_hz, np.float32),
        period_s=np.array(len(actions) / cfg.target_hz, np.float32),
        source_start_s=np.array(cfg.start_s, np.float32),
    )
    metrics = {
        "source_video": cfg.video_file,
        "source_segment_s": [cfg.start_s, cfg.start_s + len(frames) / fps],
        "source_frames": len(frames),
        "detected_frame_fraction": float((confidence.max(axis=1) > 0).mean()),
        "mean_keypoint_confidence": float(confidence[valid].mean()),
        "target_frames": len(actions),
        "target_hz": cfg.target_hz,
        "period_s": len(actions) / cfg.target_hz,
        "maximum_action_offset_rad": float(np.abs(actions).max()),
        "hip_yaw_range_rad": float(np.ptp(actions[:, [0, 9]])),
        "hip_roll_range_rad": float(np.ptp(actions[:, [1, 10]])),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Config))
