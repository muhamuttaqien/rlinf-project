"""Pure-CPU FR3 inference-input preparation tests (no ROS or model checkpoint)."""

import numpy as np
import pytest

from toolkits.realworld_check.fr3_pi05_inference_preflight import (
    audit_prediction_chunk,
    prepare_observation,
)


def measured():
    return {
        "tcp_pose": [0.3, 0.1, 0.5, 0.0, 0.0, 0.0, 1.0],
        "arm_joint_position": [0.0] * 7,
        "tcp_force": [0.0, 0.0, 0.0],
        "tcp_torque": [0.0, 0.0, 0.0],
        "robotiq_knuckle_position": [0.1],
    }


def test_prepares_rgb_two_view_and_19d_state():
    base = np.array([[[12, 21, 31]]], dtype=np.uint8)
    wrist = np.array([[[41, 52, 63]]], dtype=np.uint8)
    out = prepare_observation(measured(), base, wrist, "pick cube")
    assert out["observation/state"].shape == (19,)
    assert out["observation/state"].dtype == np.float32
    assert out["observation/image"].tolist() == [[[31, 21, 12]]]
    assert out["observation/extra_view_image"].tolist() == [[[63, 52, 41]]]
    assert out["prompt"] == "pick cube"


def test_rejects_missing_prompt_and_image():
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="prompt"):
        prepare_observation(measured(), frame, frame, "")
    with pytest.raises(ValueError, match="HWC"):
        prepare_observation(measured(), frame, np.zeros((1, 1)), "pick cube")


def test_prediction_chunk_is_audit_only():
    report = audit_prediction_chunk(np.zeros((5, 7), dtype=np.float32))
    assert report["decoded_action_shape"] == [5, 7]
    assert report["execution_approved"] is False


def test_rejects_invalid_predictions():
    with pytest.raises(ValueError, match="horizon"):
        audit_prediction_chunk(np.zeros((7,), dtype=np.float32))
    with pytest.raises(ValueError, match="Nonfinite"):
        audit_prediction_chunk(np.full((5, 7), np.nan))
