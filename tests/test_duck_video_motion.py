import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_duck_video_motion.py"
_SPEC = importlib.util.spec_from_file_location("extract_duck_video_motion", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
POINT_NAMES = _MODULE.POINT_NAMES
_fill_tracks = _MODULE._fill_tracks
retarget_tracks = _MODULE.retarget_tracks


def _synthetic_tracks(frames: int = 30) -> np.ndarray:
    base = {
        "head_left": (80, 30),
        "head_center": (100, 32),
        "head_right": (120, 34),
        "neck": (100, 65),
        "torso": (100, 90),
        "pelvis": (100, 120),
        "left_hip": (80, 122),
        "right_hip": (120, 122),
        "left_knee": (75, 160),
        "right_knee": (125, 160),
        "left_shin": (75, 190),
        "right_shin": (125, 190),
        "left_foot": (70, 220),
        "right_foot": (130, 220),
    }
    return np.asarray(
        [[base[name] for name in POINT_NAMES] for _ in range(frames)],
        dtype=np.float32,
    )


def test_fill_tracks_interpolates_occluded_point():
    tracks = _synthetic_tracks(3)
    visibility = np.ones(tracks.shape[:2], dtype=bool)
    visibility[1, 0] = False
    tracks[1, 0] = np.nan
    filled = _fill_tracks(tracks, visibility)
    assert np.allclose(filled[1, 0], filled[0, 0])


def test_retarget_tracks_uses_alternating_foot_lifts_and_knees():
    tracks = _synthetic_tracks()
    phase = np.linspace(0, 2 * np.pi, len(tracks), endpoint=False)
    tracks[:, POINT_NAMES.index("left_foot"), 1] -= 18 * np.maximum(np.sin(phase), 0)
    tracks[:, POINT_NAMES.index("right_foot"), 1] -= 18 * np.maximum(-np.sin(phase), 0)
    visibility = np.ones(tracks.shape[:2], dtype=bool)
    time, actions = retarget_tracks(tracks, visibility, source_hz=15, target_hz=50)
    assert len(time) == len(actions) == 100
    assert actions.shape[1] == 14
    assert np.isfinite(actions).all()
    assert np.ptp(actions[:, 3]) > 0.1
    assert np.ptp(actions[:, 12]) > 0.1
    assert not np.allclose(actions[:, 3], actions[:, 12])
