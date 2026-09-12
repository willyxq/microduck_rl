import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

from mjlab_microduck.imitation import DEFAULT_POSE

_SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_filmed_duck_pose.py"
_SPEC = importlib.util.spec_from_file_location("extract_filmed_duck_pose", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

analytic_retarget = _MODULE.analytic_retarget
crop_duck_panel = _MODULE.crop_duck_panel
DuckReprojectionIk = _MODULE.DuckReprojectionIk
LANDMARK_NAMES = _MODULE.LANDMARK_NAMES


def _front_landmarks(frames: int = 20) -> np.ndarray:
    points = np.zeros((frames, 5, 2), dtype=np.float32)
    points[:, 0] = (120, 70)
    points[:, 1] = (120, 80)
    points[:, 2] = (120, 170)
    points[:, 3] = (80, 250)
    points[:, 4] = (160, 250)
    return points


def test_crop_uses_right_panel_and_drops_titles():
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    frame[600:800, 700:900] = 255
    crop = crop_duck_panel(frame, 0.5, 0.26, 0.73)
    assert crop.shape[1] == 540
    assert crop.shape[0] == int(0.73 * 1920) - int(0.26 * 1920)
    assert crop[:, 160:360].sum() > 0


def test_analytic_profile_uses_root_yaw_not_tiny_offsets():
    landmarks = _front_landmarks(24)
    # Collapse foot spread and move the eye to the right: a profile turn.
    landmarks[:, 3, 0] = 118
    landmarks[:, 4, 0] = 128
    landmarks[:, 0, 0] = 150
    landmarks[:, 1, 0] = 130
    time, q, root_yaw = analytic_retarget(landmarks, source_hz=12, target_hz=50)
    assert len(time) == len(q) == len(root_yaw)
    assert q.shape[1] == 14
    assert float(np.max(np.abs(root_yaw))) > 0.8
    assert float(np.max(np.abs(q - DEFAULT_POSE))) > 0.2


def test_analytic_squat_bends_both_knees():
    landmarks = _front_landmarks()
    landmarks[8:16, 2, 1] += 22
    _, q, _ = analytic_retarget(landmarks, source_hz=12, target_hz=50)
    assert np.ptp(q[:, 3]) > 0.15
    assert np.ptp(q[:, 12]) > 0.15


def test_reprojection_ik_recovers_head_yaw():
    scene = str(Path(__file__).parents[1] / "src/mjlab_microduck/robot/microduck/scene.xml")
    solver = DuckReprojectionIk(scene)
    q_true = DEFAULT_POSE.copy()
    q_true[7] = 0.7
    solver.set_pose(DEFAULT_POSE, root_yaw=0.0)
    solver.calibrate(solver.projected_landmarks(), DEFAULT_POSE, 0.0)
    solver.set_pose(q_true, root_yaw=0.0)
    target = solver.projected_landmarks()
    q_fit, yaw_fit = solver.refine_frame(target, DEFAULT_POSE, 0.0)
    assert abs(float(q_fit[7]) - 0.7) < 0.25
    assert abs(yaw_fit) < 0.35
    assert math.isfinite(float(np.sum(q_fit)))
