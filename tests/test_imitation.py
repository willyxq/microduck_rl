from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab_microduck.imitation import (
    ACTION_DIM,
    OBS_DIM,
    BehaviorCloningPolicy,
    build_dataset,
    interpolate_actions,
    load_motion,
)

MOTION_PATH = (
    Path(__file__).parents[1] / "examples" / "imitation_learning" / "nod_sway.json"
)


def test_motion_is_periodic_and_has_deployment_shape():
    motion = load_motion(MOTION_PATH)
    assert motion.actions.shape[1] == ACTION_DIM
    assert np.array_equal(motion.actions[0], motion.actions[-1])
    assert motion.period_s == 4.0


def test_interpolation_wraps_and_hits_keyframes():
    motion = load_motion(MOTION_PATH)
    at_keys = interpolate_actions(motion, motion.phases)
    assert np.allclose(at_keys, motion.actions)
    wrapped = interpolate_actions(motion, np.array([-0.25, 0.75, 1.75]))
    assert np.allclose(wrapped[0], wrapped[1])
    assert np.allclose(wrapped[1], wrapped[2])


def test_dataset_and_policy_shapes_are_finite():
    motion = load_motion(MOTION_PATH)
    observations, actions = build_dataset(motion, sample_count=32)
    prediction = BehaviorCloningPolicy()(observations)
    assert observations.shape == (32, OBS_DIM)
    assert actions.shape == prediction.shape == (32, ACTION_DIM)
    assert torch.isfinite(prediction).all()


def test_bad_periodic_motion_is_rejected(tmp_path):
    text = MOTION_PATH.read_text(encoding="utf-8").replace(
        '"phase": 1.0,   "action": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]',
        '"phase": 1.0,   "action": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]',
    )
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="periodic"):
        load_motion(path)


def test_short_behavior_cloning_run_reduces_loss():
    torch.manual_seed(3)
    motion = load_motion(MOTION_PATH)
    observations, actions = build_dataset(motion, sample_count=512, seed=3)
    model = BehaviorCloningPolicy(hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    with torch.no_grad():
        initial = torch.nn.functional.mse_loss(model(observations), actions)
    for _ in range(40):
        loss = torch.nn.functional.mse_loss(model(observations), actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = torch.nn.functional.mse_loss(model(observations), actions)
    assert final < initial * 0.15
