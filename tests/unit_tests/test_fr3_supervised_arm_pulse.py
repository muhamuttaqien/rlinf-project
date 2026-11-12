"""Fail-closed tests for a first arm-only pulse; do not access ROS or robot."""

from types import SimpleNamespace

import pytest

from toolkits.realworld_check.fr3_supervised_arm_pulse import (
    Blocked, MAX_PULSE_S, MAX_REQUEST, NODE_NAME,
    qualify_graph, qualify_pose, validate_pulse_request,
)


def endpoint(name):
    return SimpleNamespace(node_name=name)


def test_preflight_requires_controller_and_no_spacemouse_writer():
    qualify_graph([], [endpoint("joint_impedance_ik_controller")])
    with pytest.raises(Blocked, match="Existing command publisher"):
        qualify_graph([endpoint("spacemouse_publisher")],
                      [endpoint("joint_impedance_ik_controller")])
    with pytest.raises(Blocked, match="subscriber"):
        qualify_graph([], [])


def test_active_pulse_owner_must_be_exclusive():
    qualify_graph([endpoint(NODE_NAME)],
                  [endpoint("joint_impedance_ik_controller")],
                  own_publisher=True)
    with pytest.raises(Blocked, match="Competing publisher"):
        qualify_graph([endpoint(NODE_NAME), endpoint("spacemouse_publisher")],
                      [endpoint("joint_impedance_ik_controller")],
                      own_publisher=True)


def test_pose_freshness_and_displacement():
    qualify_pose((0.3, 0.1, 0.5), receipt_monotonic=1.0, now=1.1)
    with pytest.raises(Blocked, match="Stale"):
        qualify_pose((0.3, 0.1, 0.5), receipt_monotonic=1.0, now=1.3)
    with pytest.raises(Blocked, match="Displacement"):
        qualify_pose((0.3, 0.1, 0.52), receipt_monotonic=1.0,
                     now=1.1, origin=(0.3, 0.1, 0.5))


@pytest.mark.parametrize("axis_value,pulse_seconds", [
    (0.0, 0.1), (MAX_REQUEST + 0.001, 0.1),
    (0.03, MAX_PULSE_S + 0.01), (float("nan"), 0.1),
])
def test_pulse_bounds_reject_bad_inputs(axis_value, pulse_seconds):
    with pytest.raises(Blocked):
        validate_pulse_request(axis_value, pulse_seconds)


def test_default_pulse_within_review_bounds():
    validate_pulse_request(0.03, 0.1)
