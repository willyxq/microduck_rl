#!/usr/bin/env python3
"""Extract COCO body keypoints and retarget a dance clip to MicroDuck offsets.

Run with the optional pose dependency:
  uv run --with ultralytics python scripts/extract_video_dance_motion.py ...
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
    start_s: float = 0.0
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


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.arctan2(a[:, 1] - b[:, 1], a[:, 0] - b[:, 0])


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
        raise RuntimeError("dance segment contains fewer than two frames")

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
    keypoints = gaussian_filter1d(keypoints, sigma=1.25, axis=0, mode="nearest")

    # COCO joints: nose=0, shoulders=5/6, hips=11/12, knees=13/14,
    # ankles=15/16. All features are scale-normalized before retargeting.
    shoulder_mid = 0.5 * (keypoints[:, 5] + keypoints[:, 6])
    hip_mid = 0.5 * (keypoints[:, 11] + keypoints[:, 12])
    ankle_mid = 0.5 * (keypoints[:, 15] + keypoints[:, 16])
    torso = np.linalg.norm(shoulder_mid - hip_mid, axis=1).clip(20.0)
    shoulder_width = np.linalg.norm(
        keypoints[:, 5] - keypoints[:, 6], axis=1
    ).clip(20.0)
    leg_height = (ankle_mid[:, 1] - hip_mid[:, 1]) / torso
    low, high = np.quantile(leg_height, [0.15, 0.85])
    bounce = np.clip((high - leg_height) / max(high - low, 0.1), 0.0, 1.0)
    sway = np.clip((hip_mid[:, 0] - ankle_mid[:, 0]) / torso, -1.0, 1.0)
    left_lift = np.clip(
        (keypoints[:, 16, 1] - keypoints[:, 15, 1]) / torso, 0.0, 1.2
    )
    right_lift = np.clip(
        (keypoints[:, 15, 1] - keypoints[:, 16, 1]) / torso, 0.0, 1.2
    )
    shoulder_roll = np.clip(
        _angle(keypoints[:, 6], keypoints[:, 5]), -0.45, 0.45
    )
    head_side = np.clip(
        (keypoints[:, 0, 0] - shoulder_mid[:, 0]) / shoulder_width,
        -0.8,
        0.8,
    )
    bounce_rate = np.gradient(bounce) * fps

    source_time = np.arange(len(frames), dtype=np.float32) / fps
    target_time = np.arange(
        0.0, len(frames) / fps, 1.0 / cfg.target_hz, dtype=np.float32
    )

    def sample(feature: np.ndarray) -> np.ndarray:
        return np.interp(target_time, source_time, feature).astype(np.float32)

    bounce_t = sample(bounce)
    sway_t = sample(sway)
    left_lift_t = sample(left_lift)
    right_lift_t = sample(right_lift)
    shoulder_roll_t = sample(shoulder_roll)
    head_side_t = sample(head_side)
    bounce_rate_t = sample(bounce_rate)
    actions = np.zeros((len(target_time), 14), dtype=np.float32)
    # Weight shift: same-sign hip-roll offsets lean the MicroDuck trunk.
    actions[:, 1] = 0.09 * sway_t
    actions[:, 10] = 0.09 * sway_t
    # Compact squat plus alternating lifted leg, respecting left/right mirrored
    # pitch conventions in the MicroDuck HOME pose.
    actions[:, 2] = -0.08 * bounce_t + 0.14 * left_lift_t
    actions[:, 4] = 0.08 * bounce_t - 0.09 * left_lift_t
    actions[:, 11] = 0.08 * bounce_t - 0.14 * right_lift_t
    actions[:, 13] = -0.08 * bounce_t + 0.09 * right_lift_t
    actions[:, 0] = 0.05 * left_lift_t
    actions[:, 9] = -0.05 * right_lift_t
    # MicroDuck has no arms; head motion carries some upper-body expression.
    actions[:, 6] = np.clip(0.035 * bounce_rate_t, -0.16, 0.16)
    actions[:, 7] = 0.16 * head_side_t
    actions[:, 8] = np.clip(-0.35 * shoulder_roll_t, -0.12, 0.12)
    actions = gaussian_filter1d(actions, sigma=1.0, axis=0, mode="wrap")
    # Make the extracted clip loop without a discontinuity.
    blend = min(round(0.4 * cfg.target_hz), len(actions) // 4)
    if blend > 1:
        midpoint = 0.5 * (actions[0] + actions[-1])
        ramp = np.linspace(0.0, 1.0, blend, dtype=np.float32)[:, None]
        actions[:blend] = midpoint * (1 - ramp) + actions[:blend] * ramp
        actions[-blend:] = actions[-blend:] * (1 - ramp) + midpoint * ramp

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
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Config))
