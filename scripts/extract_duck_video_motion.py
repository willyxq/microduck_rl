#!/usr/bin/env python3
"""Track a filmed MicroDuck and retarget its image motion to joint offsets.

CoTracker is optional because its checkpoint is large. Install/clone the
official implementation and download facebook/cotracker3/scaled_offline.pth,
then pass both local paths on the command line.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from scipy.ndimage import gaussian_filter1d


POINT_NAMES = (
    "head_left",
    "head_center",
    "head_right",
    "neck",
    "torso",
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_shin",
    "right_shin",
    "left_foot",
    "right_foot",
)

# Coordinates relative to the right-panel crop at source time 3.0 s. They sit
# on high-contrast physical features (shell edges, screws, and white markers).
QUERY_POINTS_NORMALIZED = np.array(
    [
        [0.37, 0.28],
        [0.55, 0.29],
        [0.75, 0.31],
        [0.56, 0.47],
        [0.56, 0.57],
        [0.56, 0.69],
        [0.36, 0.70],
        [0.78, 0.70],
        [0.34, 0.79],
        [0.80, 0.79],
        [0.38, 0.88],
        [0.77, 0.88],
        [0.38, 0.96],
        [0.77, 0.96],
    ],
    dtype=np.float32,
)


@dataclass
class Config:
    video_file: str
    output_file: str
    cotracker_repo: str
    checkpoint_file: str
    overlay_file: str | None = None
    start_s: float = 0.5
    duration_s: float = 9.5
    query_source_s: float = 3.0
    tracking_hz: float = 15.0
    target_hz: float = 50.0
    crop_left_fraction: float = 0.5
    crop_top_fraction: float = 0.15
    crop_bottom_fraction: float = 0.84
    tracking_width: int = 384
    device: str = "cuda:0"


def _fill_tracks(tracks: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    result = tracks.copy()
    timeline = np.arange(len(tracks))
    for point in range(tracks.shape[1]):
        valid = visibility[:, point] & np.isfinite(tracks[:, point]).all(axis=1)
        if valid.sum() < 2:
            raise RuntimeError(f"insufficient visible frames for {POINT_NAMES[point]}")
        for axis in range(2):
            result[:, point, axis] = np.interp(
                timeline, timeline[valid], tracks[valid, point, axis]
            )
    return result


def _center_scale(feature: np.ndarray, scale: float) -> np.ndarray:
    return (feature - np.median(feature)) / max(scale, 1e-6)


def retarget_tracks(
    tracks: np.ndarray,
    visibility: np.ndarray,
    source_hz: float,
    target_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert named 2D robot tracks into 14 MicroDuck action offsets."""
    points = {
        name: value
        for name, value in zip(POINT_NAMES, _fill_tracks(tracks, visibility).transpose(1, 0, 2))
    }
    pelvis = points["pelvis"]
    hip_mid = 0.5 * (points["left_hip"] + points["right_hip"])
    feet_mid = 0.5 * (points["left_foot"] + points["right_foot"])
    body_height = feet_mid[:, 1] - pelvis[:, 1]
    scale = float(np.median(body_height))
    if scale < 20.0:
        raise RuntimeError(f"tracked body scale is implausibly small: {scale:.1f}px")

    low_height, high_height = np.quantile(body_height, [0.1, 0.9])
    squat = np.clip(
        (high_height - body_height) / max(high_height - low_height, 0.08 * scale),
        0.0,
        1.0,
    )
    sway = np.clip(
        _center_scale(pelvis[:, 0] - feet_mid[:, 0], scale), -0.55, 0.55
    )
    trunk_side = np.clip(
        _center_scale(points["head_center"][:, 0] - hip_mid[:, 0], scale),
        -0.7,
        0.7,
    )

    left_floor = np.quantile(points["left_foot"][:, 1], 0.9)
    right_floor = np.quantile(points["right_foot"][:, 1], 0.9)
    left_lift = np.clip((left_floor - points["left_foot"][:, 1]) / (0.16 * scale), 0, 1)
    right_lift = np.clip(
        (right_floor - points["right_foot"][:, 1]) / (0.16 * scale), 0, 1
    )
    left_step = np.clip(
        _center_scale(points["left_foot"][:, 0] - points["left_hip"][:, 0], scale),
        -0.5,
        0.5,
    )
    right_step = np.clip(
        _center_scale(
            points["right_foot"][:, 0] - points["right_hip"][:, 0], scale
        ),
        -0.5,
        0.5,
    )
    head_axis = points["head_right"] - points["head_left"]
    head_roll = np.clip(
        np.arctan2(head_axis[:, 1], head_axis[:, 0])
        - np.median(np.arctan2(head_axis[:, 1], head_axis[:, 0])),
        -0.5,
        0.5,
    )
    bounce_rate = np.gradient(squat) * source_hz

    source_time = np.arange(len(tracks), dtype=np.float32) / source_hz
    target_time = np.arange(
        0.0, len(tracks) / source_hz, 1.0 / target_hz, dtype=np.float32
    )

    def sample(feature: np.ndarray) -> np.ndarray:
        return np.interp(target_time, source_time, feature).astype(np.float32)

    squat_t = sample(squat)
    sway_t = sample(sway)
    trunk_side_t = sample(trunk_side)
    left_lift_t = sample(left_lift)
    right_lift_t = sample(right_lift)
    left_step_t = sample(left_step)
    right_step_t = sample(right_step)
    head_roll_t = sample(head_roll)
    bounce_rate_t = sample(bounce_rate)

    actions = np.zeros((len(target_time), 14), dtype=np.float32)
    # Leg indices: left [yaw, roll, pitch, knee, ankle], head [5:9],
    # right [yaw, roll, pitch, knee, ankle]. Right pitch-chain signs mirror left.
    actions[:, 0] = 0.10 * left_step_t
    actions[:, 1] = 0.16 * sway_t
    actions[:, 2] = -0.14 * squat_t + 0.20 * left_lift_t
    actions[:, 3] = -0.22 * squat_t - 0.24 * left_lift_t
    actions[:, 4] = 0.14 * squat_t - 0.08 * left_lift_t
    actions[:, 9] = -0.10 * right_step_t
    actions[:, 10] = 0.16 * sway_t
    actions[:, 11] = 0.14 * squat_t - 0.20 * right_lift_t
    actions[:, 12] = 0.22 * squat_t + 0.24 * right_lift_t
    actions[:, 13] = -0.14 * squat_t + 0.08 * right_lift_t
    actions[:, 6] = np.clip(0.04 * bounce_rate_t, -0.14, 0.14)
    actions[:, 7] = 0.26 * trunk_side_t
    actions[:, 8] = 0.28 * head_roll_t + 0.10 * sway_t
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
    import torch

    capture = cv2.VideoCapture(cfg.video_file)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_MSEC, cfg.start_s * 1000.0)
    step = source_fps / cfg.tracking_hz
    source_frames = round(cfg.duration_s * source_fps)
    wanted = {round(index * step) for index in range(round(cfg.duration_s * cfg.tracking_hz))}
    x0 = round(width * cfg.crop_left_fraction)
    y0 = round(height * cfg.crop_top_fraction)
    y1 = round(height * cfg.crop_bottom_fraction)
    resized_height = round((y1 - y0) * cfg.tracking_width / (width - x0))
    frames = []
    for index in range(source_frames):
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            crop = frame[y0:y1, x0:width]
            frames.append(
                cv2.resize(crop, (cfg.tracking_width, resized_height))
            )
    capture.release()
    if len(frames) < 2:
        raise RuntimeError("video segment contains fewer than two tracking frames")

    sys.path.insert(0, str(Path(cfg.cotracker_repo).resolve()))
    from cotracker.predictor import CoTrackerPredictor

    model = CoTrackerPredictor(
        checkpoint=cfg.checkpoint_file, offline=True, window_len=60
    ).to(cfg.device)
    video = (
        torch.from_numpy(np.stack(frames))
        .permute(0, 3, 1, 2)[None]
        .float()
        .to(cfg.device)
    )
    query_frame = round((cfg.query_source_s - cfg.start_s) * cfg.tracking_hz)
    query_xy = QUERY_POINTS_NORMALIZED * np.array(
        [cfg.tracking_width, resized_height], dtype=np.float32
    )
    queries = np.column_stack(
        [np.full(len(query_xy), query_frame, dtype=np.float32), query_xy]
    )
    with torch.inference_mode():
        predicted, visible = model(
            video,
            queries=torch.from_numpy(queries)[None].to(cfg.device),
            backward_tracking=True,
        )
    tracks = predicted[0].cpu().numpy()
    visibility = visible[0].cpu().numpy()
    target_time, actions = retarget_tracks(
        tracks, visibility, cfg.tracking_hz, cfg.target_hz
    )

    output = Path(cfg.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        action_offsets=actions,
        target_time_s=target_time,
        source_tracks=tracks,
        source_visibility=visibility,
        point_names=np.asarray(POINT_NAMES),
        source_fps=np.array(source_fps, np.float32),
        tracking_hz=np.array(cfg.tracking_hz, np.float32),
        target_hz=np.array(cfg.target_hz, np.float32),
        period_s=np.array(len(actions) / cfg.target_hz, np.float32),
        source_start_s=np.array(cfg.start_s, np.float32),
    )

    if cfg.overlay_file:
        colors = [
            (255, 80, 80),
            (80, 255, 80),
            (80, 80, 255),
            (255, 255, 80),
        ]
        writer = cv2.VideoWriter(
            cfg.overlay_file,
            cv2.VideoWriter_fourcc(*"mp4v"),
            cfg.tracking_hz,
            (cfg.tracking_width, resized_height),
        )
        for frame_index, frame in enumerate(frames):
            annotated = frame.copy()
            for point_index, (x, y) in enumerate(tracks[frame_index]):
                color = colors[point_index % len(colors)]
                cv2.circle(annotated, (round(x), round(y)), 5, color, -1)
                cv2.putText(
                    annotated,
                    POINT_NAMES[point_index],
                    (round(x) + 5, round(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            writer.write(annotated)
        writer.release()

    metrics = {
        "source_video": cfg.video_file,
        "source_segment_s": [cfg.start_s, cfg.start_s + len(frames) / cfg.tracking_hz],
        "tracking_frames": len(frames),
        "tracking_hz": cfg.tracking_hz,
        "mean_point_visibility": float(visibility.mean()),
        "minimum_point_visibility": float(visibility.mean(axis=0).min()),
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
