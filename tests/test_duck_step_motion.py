import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_duck_step_motion.py"
_SPEC = importlib.util.spec_from_file_location("extract_duck_step_motion", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
retarget_duck_step = _MODULE.retarget_duck_step


def _standing_keypoints(frames: int = 40) -> np.ndarray:
    keypoints = np.zeros((frames, 17, 2), dtype=np.float32)
    keypoints[:, 0] = (100, 20)
    keypoints[:, 5] = (80, 60)
    keypoints[:, 6] = (120, 60)
    keypoints[:, 11] = (85, 140)
    keypoints[:, 12] = (115, 140)
    keypoints[:, 13] = (80, 190)
    keypoints[:, 14] = (120, 190)
    keypoints[:, 15] = (75, 240)
    keypoints[:, 16] = (125, 240)
    return keypoints


def test_duck_step_retarget_uses_opposite_hip_yaw_for_sway():
    keypoints = _standing_keypoints()
    phase = np.linspace(0, 2 * np.pi, len(keypoints), endpoint=False)
    keypoints[:, 11, 0] += 28 * np.sin(phase)
    keypoints[:, 12, 0] += 28 * np.sin(phase)
    time, actions = retarget_duck_step(keypoints, source_fps=20, target_hz=50)
    assert len(time) == len(actions)
    assert actions.shape[1] == 14
    assert np.ptp(actions[:, 0]) > 0.06
    assert np.ptp(actions[:, 9]) > 0.06
    assert np.corrcoef(actions[:, 0], actions[:, 9])[0, 1] < -0.5


def test_duck_step_retarget_uses_same_sign_hip_roll_for_weight_shift():
    keypoints = _standing_keypoints()
    phase = np.linspace(0, 2 * np.pi, len(keypoints), endpoint=False)
    keypoints[:, 11, 0] += 16 * np.sin(phase)
    keypoints[:, 12, 0] += 16 * np.sin(phase)
    _, actions = retarget_duck_step(keypoints, source_fps=20, target_hz=50)
    assert np.corrcoef(actions[:, 1], actions[:, 10])[0, 1] > 0.5
    assert np.ptp(actions[:, 1]) > 0.05
