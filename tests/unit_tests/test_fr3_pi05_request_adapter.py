"""Pure request-adapter checks; no ROS or physical motor access."""

import pytest

from toolkits.realworld_check.fr3_pi05_request_adapter import (
    ARM_AXES, GripperRequestFilter, identity_arm_request, validate_action_chunk,
)


def test_axis_names_and_identity_no_gain():
    assert ARM_AXES == ("linear_x", "linear_y", "linear_z",
                        "angular_x", "angular_y", "angular_z")
    row = [0.5, -0.1, 0.25, -0.2, 0.3, -0.4, 1.0]
    assert identity_arm_request(row) == tuple(row[:6])
    assert validate_action_chunk([row for _ in range(10)])[0] == row


@pytest.mark.parametrize("bad", [
    [[0.0] * 7] * 9,
    [[1.00001, 0, 0, 0, 0, 0, 1.0]] * 10,
    [[0, 0, 0, 0, 0, 0, 1.1]] * 10,
    [[0, 0, 0, 0, 0, 0, float("nan")]] * 10,
])
def test_invalid_prediction_never_clipped(bad):
    with pytest.raises(ValueError):
        validate_action_chunk(bad)


def test_debounced_gripper_suppresses_noise():
    g = GripperRequestFilter(confirmations=3, min_goal_interval_s=0.8)
    assert g.step(1.0, now=0.0) == "hold"
    assert g.step(0.45, now=0.1) == "hold"
    assert g.step(1.0, now=0.2) == "hold"
    assert g.step(1.0, now=0.3) == "hold"
    assert g.step(1.0, now=0.4) == "open"
    assert g.step(0.0, now=0.5) == "hold"
    assert g.step(0.0, now=0.6) == "hold"
    assert g.step(0.0, now=0.7) == "hold"  # no repeated goal within 0.8s
    assert g.step(0.0, now=1.3) == "close"


def test_reset_removes_old_request_state():
    g = GripperRequestFilter()
    for t in (0.0, 0.1, 0.2):
        g.step(1.0, now=t)
    g.reset()
    assert g.step(0.0, now=0.4) == "hold"
