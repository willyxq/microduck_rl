#!/usr/bin/env python3
"""Reconstruct MicroDuck joints from the filmed Korean duck-dance clip.

Previous CoTracker queries sat on the Bilibili caption, so the feet never
moved and the teacher stayed a standing wiggle. This script:

1. Crops only the right-panel duck (no titles).
2. Detects the camera-eye, pelvis, and orange feet from appearance.
3. Maps those landmarks to joints with a closed-form geometric retarget.
4. Optionally refines the pose with MuJoCo 2D-reprojection IK (RoboPose-lite).

The 90° turn in the source is whole-body root yaw, not a 14-DoF offset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from scipy.ndimage import gaussian_filter1d

from mjlab_microduck.imitation import DEFAULT_POSE, JOINT_NAMES

LANDMARK_NAMES = ("eye", "head", "pelvis", "left_foot", "right_foot")
JOINT_LIMITS = np.array(
    [
        [-0.436, 0.524],
        [-0.384, 0.384],
        [-1.571, 1.571],
        [-1.571, 1.571],
        [-1.571, 1.571],
        [-1.571, 1.047],
        [-1.571, 1.571],
        [-2.967, 2.967],
        [-0.436, 0.436],
        [-0.524, 0.436],
        [-0.384, 0.384],
        [-1.571, 1.571],
        [-1.571, 1.571],
        [-1.571, 1.571],
    ],
    dtype=np.float32,
)


@dataclass
class Config:
    video_file: str
    output_file: str
    overlay_file: str | None = None
    start_s: float = 0.4
    duration_s: float = 7.6
    target_hz: float = 50.0
    method: str = "ik"
    crop_left_fraction: float = 0.5
    crop_top_fraction: float = 0.26
    crop_bottom_fraction: float = 0.73
    scene_file: str = "src/mjlab_microduck/robot/microduck/scene.xml"


def crop_duck_panel(
    frame: np.ndarray,
    left_fraction: float,
    top_fraction: float,
    bottom_fraction: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    return frame[
        int(height * top_fraction) : int(height * bottom_fraction),
        int(width * left_fraction) :,
    ]


def _components(mask: np.ndarray, min_area: int) -> list[tuple[float, np.ndarray, np.ndarray]]:
    import cv2

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    found = []
    for index in range(1, num):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == index
        found.append((area, centroids[index, :2].astype(np.float32), component))
    found.sort(key=lambda item: item[0], reverse=True)
    return found


def detect_landmarks(bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Find the filmed duck's eye, head, pelvis and orange feet."""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    height, width = bgr.shape[:2]
    orange = cv2.medianBlur(cv2.inRange(hsv, (3, 90, 110), (20, 255, 255)), 5)
    cream = cv2.medianBlur(cv2.inRange(hsv, (0, 0, 185), (30, 55, 255)), 5)
    led = cv2.medianBlur(cv2.inRange(hsv, (28, 60, 140), (70, 255, 255)), 3)

    bill_band = np.zeros_like(orange)
    bill_band[int(0.08 * height) : int(0.50 * height)] = 255
    foot_band = np.zeros_like(orange)
    foot_band[int(0.86 * height) : int(0.98 * height)] = 255
    hip_band = np.zeros_like(cream)
    hip_band[int(0.40 * height) : int(0.70 * height)] = 255

    # The orange bill is the most stable head anchor; the bright monitor is not.
    bill_hits = _components(cv2.bitwise_and(orange, bill_band), min_area=180)
    if not bill_hits:
        bill_hits = _components(cv2.bitwise_and(orange, bill_band), min_area=80)
    bill = bill_hits[0][1] if bill_hits else np.array([0.5 * width, 0.28 * height], np.float32)
    axis_x = float(bill[0])

    head_window = np.zeros_like(cream)
    x0 = int(np.clip(axis_x - 0.22 * width, 0, width - 1))
    x1 = int(np.clip(axis_x + 0.22 * width, 0, width))
    y0 = int(np.clip(bill[1] - 0.22 * height, 0, height - 1))
    y1 = int(np.clip(bill[1] + 0.08 * height, 0, height))
    head_window[y0:y1, x0:x1] = 255
    head_hits = _components(cv2.bitwise_and(cream, head_window), min_area=200)
    head = head_hits[0][1] if head_hits else bill.copy()

    eye_search = np.zeros_like(led)
    eye_search[
        max(0, y0 - int(0.06 * height)) : min(height, y1 + int(0.10 * height)),
        max(0, x0 - int(0.08 * width)) : min(width, x1 + int(0.18 * width)),
    ] = 255
    eye_hits = _components(cv2.bitwise_and(led, eye_search), min_area=6)
    eye = eye_hits[0][1] if eye_hits else head.copy()

    pelvis_hits = [
        item
        for item in _components(cv2.bitwise_and(cream, hip_band), min_area=180)
        if abs(float(item[1][0]) - axis_x) < 0.20 * width
    ]
    pelvis = (
        pelvis_hits[0][1]
        if pelvis_hits
        else np.array([axis_x, 0.56 * height], np.float32)
    )

    foot_hits = [
        item
        for item in _components(cv2.bitwise_and(orange, foot_band), min_area=60)
        if abs(float(item[1][0]) - axis_x) < 0.42 * width
    ]
    feet = [item[1] for item in foot_hits[:2]]
    feet.sort(key=lambda point: point[0])
    if len(feet) == 0:
        left_foot = np.array([axis_x - 0.16 * width, 0.90 * height], np.float32)
        right_foot = np.array([axis_x + 0.16 * width, 0.90 * height], np.float32)
    elif len(feet) == 1:
        left_foot = feet[0]
        right_foot = np.array([2.0 * axis_x - feet[0][0], feet[0][1]], np.float32)
        if right_foot[0] < left_foot[0]:
            left_foot, right_foot = right_foot, left_foot
    else:
        left_foot, right_foot = feet[0], feet[1]

    return {
        "eye": eye,
        "head": head,
        "pelvis": pelvis,
        "left_foot": left_foot,
        "right_foot": right_foot,
    }


def detect_video_landmarks(
    frames: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.full((len(frames), len(LANDMARK_NAMES), 2), np.nan, dtype=np.float32)
    for index, frame in enumerate(frames):
        detected = detect_landmarks(frame)
        for landmark_index, name in enumerate(LANDMARK_NAMES):
            raw[index, landmark_index] = detected[name]
    visibility = np.isfinite(raw).all(axis=2)
    filled = raw.copy()
    timeline = np.arange(len(frames))
    for landmark_index in range(len(LANDMARK_NAMES)):
        valid = visibility[:, landmark_index]
        if valid.sum() < 3:
            raise RuntimeError(
                f"insufficient detections for {LANDMARK_NAMES[landmark_index]}"
            )
        for axis in range(2):
            filled[:, landmark_index, axis] = np.interp(
                timeline, timeline[valid], raw[valid, landmark_index, axis]
            )
    return (
        gaussian_filter1d(filled, sigma=1.2, axis=0, mode="nearest"),
        visibility,
    )


def _center_scale(values: np.ndarray, scale: float) -> np.ndarray:
    return (values - np.median(values)) / max(scale, 1e-6)


def analytic_retarget(
    landmarks: np.ndarray,
    source_hz: float,
    target_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form joints + root yaw from 2D duck landmarks."""
    points = {
        name: landmarks[:, index]
        for index, name in enumerate(LANDMARK_NAMES)
    }
    feet_mid = 0.5 * (points["left_foot"] + points["right_foot"])
    body_height = feet_mid[:, 1] - points["pelvis"][:, 1]
    scale = float(np.median(np.abs(body_height)))
    if scale < 15.0:
        raise RuntimeError(f"filmed duck scale is implausibly small: {scale:.1f}px")

    foot_spread = np.abs(points["left_foot"][:, 0] - points["right_foot"][:, 0]) / scale
    eye_offset = (points["eye"][:, 0] - points["head"][:, 0]) / scale
    # Front-on the feet are ~0.7 body-heights apart; profile collapses that
    # and the camera-eye sits on one end of the head.
    view = np.maximum(
        np.clip((0.70 - foot_spread) / 0.50, 0.0, 1.0),
        np.clip(np.abs(eye_offset) / 0.28, 0.0, 1.0),
    )
    yaw_sign = float(np.sign(np.nanmedian(eye_offset)))
    if yaw_sign == 0.0:
        yaw_sign = 1.0
    root_yaw = yaw_sign * view * (0.5 * np.pi)

    squat = np.clip(
        (np.quantile(body_height, 0.9) - body_height)
        / max(np.ptp(np.quantile(body_height, [0.1, 0.9])), 0.08 * scale),
        0.0,
        1.2,
    )
    sway = np.clip(
        _center_scale(points["pelvis"][:, 0] - feet_mid[:, 0], scale), -0.7, 0.7
    )
    head_pitch = np.clip(
        _center_scale(points["head"][:, 1] - points["pelvis"][:, 1], scale),
        -0.8,
        0.8,
    )
    head_roll = np.clip(
        _center_scale(points["head"][:, 0] - points["pelvis"][:, 0], scale),
        -0.5,
        0.5,
    )
    left_lift = np.clip(
        (
            np.quantile(points["left_foot"][:, 1], 0.9)
            - points["left_foot"][:, 1]
        )
        / (0.18 * scale),
        0.0,
        1.0,
    )
    right_lift = np.clip(
        (
            np.quantile(points["right_foot"][:, 1], 0.9)
            - points["right_foot"][:, 1]
        )
        / (0.18 * scale),
        0.0,
        1.0,
    )
    # Residual head yaw after removing the whole-body turn.
    head_yaw = np.clip(1.10 * eye_offset - 0.55 * root_yaw, -1.2, 1.2)

    source_time = np.arange(len(landmarks), dtype=np.float32) / source_hz
    target_time = np.arange(
        0.0, len(landmarks) / source_hz, 1.0 / target_hz, dtype=np.float32
    )

    def sample(feature: np.ndarray) -> np.ndarray:
        return np.interp(target_time, source_time, feature).astype(np.float32)

    squat_t = sample(squat)
    sway_t = sample(sway)
    head_pitch_t = sample(head_pitch)
    head_roll_t = sample(head_roll)
    head_yaw_t = sample(head_yaw)
    root_yaw_t = sample(root_yaw)
    left_lift_t = sample(left_lift)
    right_lift_t = sample(right_lift)

    q = np.repeat(DEFAULT_POSE[None, :], len(target_time), axis=0).astype(np.float32)
    q[:, 0] = 0.12 * sway_t
    q[:, 1] = DEFAULT_POSE[1] + 0.22 * sway_t
    q[:, 2] = DEFAULT_POSE[2] - 0.42 * squat_t + 0.18 * left_lift_t
    q[:, 3] = DEFAULT_POSE[3] - 0.55 * squat_t - 0.20 * left_lift_t
    q[:, 4] = DEFAULT_POSE[4] + 0.28 * squat_t - 0.10 * left_lift_t
    q[:, 5] = DEFAULT_POSE[5] + 0.18 * head_pitch_t - 0.10 * squat_t
    q[:, 6] = DEFAULT_POSE[6] + 0.35 * head_pitch_t
    q[:, 7] = head_yaw_t
    q[:, 8] = np.clip(0.35 * head_roll_t, *JOINT_LIMITS[8])
    q[:, 9] = -0.12 * sway_t
    q[:, 10] = DEFAULT_POSE[10] + 0.22 * sway_t
    q[:, 11] = DEFAULT_POSE[11] + 0.42 * squat_t - 0.18 * right_lift_t
    q[:, 12] = DEFAULT_POSE[12] + 0.55 * squat_t + 0.20 * right_lift_t
    q[:, 13] = DEFAULT_POSE[13] - 0.28 * squat_t + 0.10 * right_lift_t
    q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    q = gaussian_filter1d(q, sigma=1.0, axis=0, mode="nearest")
    root_yaw_t = gaussian_filter1d(root_yaw_t, sigma=1.2, mode="nearest")
    return target_time, q, root_yaw_t.astype(np.float32)


def _yaw_quaternion(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


class DuckReprojectionIk:
    """Weak-perspective 2D-to-q fit on the MicroDuck MuJoCo model."""

    def __init__(self, scene_file: str) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(scene_file)
        self.data = mujoco.MjData(self.model)
        self.joint_qpos = self.model.jnt_qposadr[self.model.actuator_trnid[:, 0]]
        free = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint"
        )
        self.free_qpos = int(self.model.jnt_qposadr[free])
        self.site_ids = {
            "eye": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "head_camera"
            ),
            "left_foot": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot"
            ),
            "right_foot": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot"
            ),
        }
        self.body_ids = {
            "head": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft"
            ),
            "pelvis": mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base"
            ),
        }
        self.camera = {"s": 1800.0, "cx": 270.0, "cy": 450.0}

    def set_pose(self, q: np.ndarray, root_yaw: float, root_z: float = 0.12) -> None:
        self.data.qpos[self.free_qpos : self.free_qpos + 3] = [0.0, 0.0, root_z]
        self.data.qpos[self.free_qpos + 3 : self.free_qpos + 7] = _yaw_quaternion(
            float(root_yaw)
        )
        self.data.qpos[self.joint_qpos] = q
        self.mujoco.mj_forward(self.model, self.data)

    def world_points(self) -> dict[str, np.ndarray]:
        points = {}
        for name, site_id in self.site_ids.items():
            points[name] = self.data.site_xpos[site_id].copy()
        for name, body_id in self.body_ids.items():
            points[name] = self.data.xpos[body_id].copy()
        return points

    def project(self, world: np.ndarray) -> np.ndarray:
        # Front camera looking along -X: image x ← -Y, image y ← -Z.
        return np.array(
            [
                self.camera["cx"] - self.camera["s"] * world[1],
                self.camera["cy"] - self.camera["s"] * world[2],
            ],
            dtype=np.float32,
        )

    def projected_landmarks(self) -> np.ndarray:
        world = self.world_points()
        return np.stack([self.project(world[name]) for name in LANDMARK_NAMES])

    def calibrate(self, landmarks: np.ndarray, q: np.ndarray, root_yaw: float) -> None:
        self.set_pose(q, root_yaw)
        world = self.world_points()
        used = []
        observed = []
        for name in ("eye", "pelvis", "left_foot", "right_foot"):
            image = landmarks[LANDMARK_NAMES.index(name)]
            if not np.isfinite(image).all():
                continue
            used.append(world[name])
            observed.append(image)
        used_xy = np.stack([[-p[1], -p[2]] for p in used])
        observed_xy = np.stack(observed)
        scale = float(
            np.linalg.norm(observed_xy - observed_xy.mean(0))
            / max(np.linalg.norm(used_xy - used_xy.mean(0)), 1e-6)
        )
        offset = observed_xy.mean(0) - scale * used_xy.mean(0)
        self.camera = {"s": scale, "cx": float(offset[0]), "cy": float(offset[1])}

    def refine_frame(
        self,
        target: np.ndarray,
        q0: np.ndarray,
        yaw0: float,
    ) -> tuple[np.ndarray, float]:
        def error(q: np.ndarray, yaw: float) -> float:
            self.set_pose(q, yaw)
            projected = self.projected_landmarks()
            valid = np.isfinite(target).all(axis=1)
            if not valid.any():
                return 1e6
            return float(np.mean(np.linalg.norm(projected[valid] - target[valid], axis=1)))

        best_q = np.clip(q0, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]).astype(np.float32)
        best_yaw = float(yaw0)
        best = error(best_q, best_yaw)
        for head_yaw in np.linspace(-1.2, 1.2, 17):
            trial = best_q.copy()
            trial[7] = head_yaw
            score = error(trial, best_yaw)
            if score < best:
                best_q, best = trial, score
        for root_yaw in np.linspace(best_yaw - 0.8, best_yaw + 0.8, 13):
            score = error(best_q, float(root_yaw))
            if score < best:
                best_yaw, best = float(root_yaw), score
        return best_q, best_yaw


def ik_retarget(
    landmarks: np.ndarray,
    source_hz: float,
    target_hz: float,
    scene_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    time, q_analytic, yaw_analytic = analytic_retarget(
        landmarks, source_hz, target_hz
    )
    solver = DuckReprojectionIk(scene_file)
    calib_index = int(np.argmin(np.abs(yaw_analytic)))
    calib_source = int(
        np.clip(round(time[calib_index] * source_hz), 0, len(landmarks) - 1)
    )
    solver.calibrate(landmarks[calib_source], q_analytic[calib_index], 0.0)
    q = q_analytic.copy()
    yaw = yaw_analytic.copy()
    mean_error = 0.0
    for index in range(len(time)):
        source_index = int(
            np.clip(round(time[index] * source_hz), 0, len(landmarks) - 1)
        )
        q[index], yaw[index] = solver.refine_frame(
            landmarks[source_index], q[index], yaw[index]
        )
        solver.set_pose(q[index], yaw[index])
        projected = solver.projected_landmarks()
        mean_error += float(
            np.nanmean(np.linalg.norm(projected - landmarks[source_index], axis=1))
        )
        if index + 1 < len(time):
            q[index + 1] = 0.65 * q[index] + 0.35 * q[index + 1]
            yaw[index + 1] = 0.65 * yaw[index] + 0.35 * yaw[index + 1]
    q = gaussian_filter1d(q, sigma=0.8, axis=0, mode="nearest")
    yaw = gaussian_filter1d(yaw, sigma=1.0, mode="nearest")
    return time, q, yaw.astype(np.float32), mean_error / max(len(time), 1)


def _loop_blend(values: np.ndarray, hz: float) -> np.ndarray:
    blend = min(round(0.30 * hz), len(values) // 4)
    if blend <= 1:
        return values
    midpoint = 0.5 * (values[0] + values[-1])
    ramp = np.linspace(0.0, 1.0, blend, dtype=np.float32)
    if values.ndim == 2:
        ramp = ramp[:, None]
    values = values.copy()
    values[:blend] = midpoint * (1 - ramp) + values[:blend] * ramp
    values[-blend:] = values[-blend:] * (1 - ramp) + midpoint * ramp
    return values


def load_video_crops(cfg: Config) -> tuple[list[np.ndarray], float]:
    import cv2

    capture = cv2.VideoCapture(cfg.video_file)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.set(cv2.CAP_PROP_POS_MSEC, cfg.start_s * 1000.0)
    frames = []
    for _ in range(round(cfg.duration_s * fps)):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(
            crop_duck_panel(
                frame,
                cfg.crop_left_fraction,
                cfg.crop_top_fraction,
                cfg.crop_bottom_fraction,
            )
        )
    capture.release()
    if len(frames) < 8:
        raise RuntimeError("filmed duck segment is too short")
    return frames, fps


def write_overlay(
    frames: list[np.ndarray],
    landmarks: np.ndarray,
    path: str,
    fps: float,
) -> None:
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    colors = {
        "eye": (0, 255, 255),
        "head": (255, 180, 0),
        "pelvis": (0, 255, 0),
        "left_foot": (255, 80, 80),
        "right_foot": (80, 80, 255),
    }
    for frame, points in zip(frames, landmarks):
        annotated = frame.copy()
        for name, (x, y) in zip(LANDMARK_NAMES, points):
            if not np.isfinite([x, y]).all():
                continue
            cv2.circle(annotated, (int(x), int(y)), 7, colors[name], -1)
            cv2.putText(
                annotated,
                name,
                (int(x) + 6, int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colors[name],
                1,
                cv2.LINE_AA,
            )
        writer.write(annotated)
    writer.release()


def main(cfg: Config) -> None:
    if cfg.method not in {"analytic", "ik"}:
        raise ValueError("method must be analytic or ik")
    frames, fps = load_video_crops(cfg)
    landmarks, visibility = detect_video_landmarks(frames)
    if cfg.method == "analytic":
        time, q, root_yaw = analytic_retarget(landmarks, fps, cfg.target_hz)
        mean_reprojection_px = float("nan")
    else:
        time, q, root_yaw, mean_reprojection_px = ik_retarget(
            landmarks, fps, cfg.target_hz, cfg.scene_file
        )
    q = _loop_blend(q, cfg.target_hz)
    root_yaw = _loop_blend(root_yaw, cfg.target_hz)
    offsets = (q - DEFAULT_POSE).astype(np.float32)

    output = Path(cfg.output_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        action_offsets=offsets,
        joint_pos=q,
        root_yaw=root_yaw,
        target_time_s=time,
        source_landmarks=landmarks,
        source_visibility=visibility,
        landmark_names=np.asarray(LANDMARK_NAMES),
        joint_names=np.asarray(JOINT_NAMES),
        source_fps=np.array(fps, np.float32),
        target_hz=np.array(cfg.target_hz, np.float32),
        period_s=np.array(len(offsets) / cfg.target_hz, np.float32),
        source_start_s=np.array(cfg.start_s, np.float32),
        method=np.asarray(cfg.method),
    )
    if cfg.overlay_file:
        write_overlay(frames, landmarks, cfg.overlay_file, fps)
    metrics = {
        "source_video": cfg.video_file,
        "method": cfg.method,
        "source_segment_s": [cfg.start_s, cfg.start_s + len(frames) / fps],
        "source_frames": len(frames),
        "mean_landmark_visibility": float(visibility.mean()),
        "minimum_landmark_visibility": float(visibility.mean(axis=0).min()),
        "target_frames": len(offsets),
        "period_s": len(offsets) / cfg.target_hz,
        "maximum_action_offset_rad": float(np.abs(offsets).max()),
        "head_yaw_range_rad": float(np.ptp(q[:, 7])),
        "root_yaw_range_rad": float(np.ptp(root_yaw)),
        "hip_pitch_range_rad": float(np.ptp(q[:, [2, 11]])),
        "mean_reprojection_px": mean_reprojection_px,
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Config))
