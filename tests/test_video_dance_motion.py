import importlib.util
import sys
from pathlib import Path

import numpy as np

_SCRIPT = Path(__file__).parents[1] / "scripts" / "extract_video_dance_motion.py"
_SPEC = importlib.util.spec_from_file_location("extract_video_dance_motion", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_fill_missing = _MODULE._fill_missing


def test_fill_missing_keypoints_interpolates_time_axis():
    values = np.array(
        [
            [[0.0, 10.0]],
            [[np.nan, np.nan]],
            [[2.0, 14.0]],
        ],
        dtype=np.float32,
    )
    valid = np.array([[True], [False], [True]])
    filled = _fill_missing(values, valid)
    assert np.allclose(filled[:, 0], [[0, 10], [1, 12], [2, 14]])


def test_fill_missing_rejects_unrecoverable_joint():
    values = np.zeros((3, 1, 2), dtype=np.float32)
    valid = np.array([[False], [True], [False]])
    try:
        _fill_missing(values, valid)
    except RuntimeError as error:
        assert "insufficient detections" in str(error)
    else:
        raise AssertionError("expected insufficient keypoints to fail")
