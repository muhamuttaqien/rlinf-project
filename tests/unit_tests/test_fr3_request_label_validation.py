"""Pure, read-only checks for FR3 request-action label diagnostics."""

import numpy as np
import pytest

from toolkits.realworld_check.fr3_request_label_validation import (
    _assert_same_reconstruction,
    _corr,
    _relative_world_rotvec,
)


def test_world_rotation_identity():
    unit = np.array([0., 0., 0., 1.])
    assert np.allclose(_relative_world_rotvec(unit, unit), np.zeros(3))


def test_world_rotation_z_positive():
    unit = np.array([0., 0., 0., 1.])
    angle = 0.2
    changed = np.array([0., 0., np.sin(angle / 2), np.cos(angle / 2)])
    assert np.allclose(
        _relative_world_rotvec(unit, changed), [0, 0, angle], atol=1e-8
    )


def test_weak_or_constant_signals_do_not_claim_alignment():
    assert _corr([1.] * 40, list(range(40))) is None
    assert _corr(list(range(10)), list(range(10))) is None


def test_reconstruction_must_match_original_labels():
    original = [{
        "frame_index": 0,
        "observation_camera_receipt_ns": 100,
        "next_observation_camera_receipt_ns": 200,
        "candidate_complete": True,
        "gripper_request_open_fraction": 1.0,
        "state19_candidate": [0.] * 19,
        "spacemouse_mean_request_6d": [0.1] * 6,
        "candidate_request_action_7d": [0.1] * 6 + [1.0],
    }]
    _assert_same_reconstruction(original, [original[0].copy()], "episode_000")
    altered = [dict(original[0], candidate_request_action_7d=[0.2] * 6 + [1.])]
    with pytest.raises(ValueError, match="candidate_request_action_7d"):
        _assert_same_reconstruction(original, altered, "episode_000")
