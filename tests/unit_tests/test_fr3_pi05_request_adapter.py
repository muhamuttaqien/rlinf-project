# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pure request-adapter checks; no ROS or physical motor access."""

import pytest

from toolkits.realworld_check.fr3_pi05_request_adapter import (
    ARM_AXES,
    GripperRequestFilter,
    identity_arm_request,
    validate_action_chunk,
)


def test_axis_names_and_identity_no_gain():
    assert ARM_AXES == (
        "linear_x",
        "linear_y",
        "linear_z",
        "angular_x",
        "angular_y",
        "angular_z",
    )
    row = [0.5, -0.1, 0.25, -0.2, 0.3, -0.4, 1.0]
    assert identity_arm_request(row) == tuple(row[:6])
    assert validate_action_chunk([row for _ in range(10)])[0] == row


@pytest.mark.parametrize(
    "bad",
    [
        [[0.0] * 7] * 9,
        [[1.00001, 0, 0, 0, 0, 0, 1.0]] * 10,
        [[0, 0, 0, 0, 0, 0, 1.1]] * 10,
        [[0, 0, 0, 0, 0, 0, float("nan")]] * 10,
    ],
)
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


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.0, 0.0),
        (0.30, 0.30),
        (0.5, 0.5),
        (0.70, 0.70),
        (1.0, 1.0),
        (-0.001, 0.0),
        (-0.05276369364422559, 0.0),
        (-0.05716913694965839, 0.0),
        (-0.06, 0.0),
        (1.001, 1.0),
        (1.05716913694965839, 1.0),
        (1.06, 1.0),
    ],
)
def test_gripper_saturation_preserves_arm_and_input(raw, expected):
    import math

    arm = [-1.0, -0.0, 0.125, -0.25, 0.5, 1.0]
    chunk = [arm + [raw] for _ in range(10)]
    adapted = validate_action_chunk(chunk)
    assert adapted == [arm + [expected] for _ in range(10)]
    assert chunk == [arm + [raw] for _ in range(10)]
    assert math.copysign(1.0, adapted[0][1]) == -1.0
    assert identity_arm_request(chunk[0]) == tuple(arm)
    assert validate_action_chunk(adapted) == adapted


@pytest.mark.parametrize("channel", range(7))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_prediction_rejects_entire_chunk(channel, value):
    chunk = [[0.0] * 7 for _ in range(10)]
    chunk[-1][channel] = value
    with pytest.raises(ValueError, match="nonfinite"):
        validate_action_chunk(chunk)


@pytest.mark.parametrize("value", [-0.0600000001, 1.0600000001, -0.1, 1.1, -1.0, 2.0])
def test_gripper_saturation_rejects_larger_errors(value):
    with pytest.raises(ValueError, match="latched_gripper_open_request"):
        validate_action_chunk([[0.0] * 6 + [value] for _ in range(10)])


@pytest.mark.parametrize("channel", range(6))
@pytest.mark.parametrize("value", [-1.000001, 1.000001])
def test_gripper_saturation_never_relaxes_arm_limits(channel, value):
    chunk = [[0.0] * 6 + [-0.057] for _ in range(10)]
    chunk[-1][channel] = value
    with pytest.raises(ValueError, match=ARM_AXES[channel]):
        validate_action_chunk(chunk)


def test_saturation_preserves_gripper_thresholds_debounce_and_goal_interval():
    requests = [1.04, 0.69, 0.70, 1.06, 1.02, 1.01, -0.057, 0.30, -0.06, 0.0]
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.3]
    chunk = [[0.0] * 6 + [g] for g in requests]
    adapted = validate_action_chunk(chunk)
    gripper = GripperRequestFilter()
    assert [gripper.step(row[6], now=t) for row, t in zip(adapted, times)] == [
        "hold",
        "hold",
        "hold",
        "hold",
        "open",
        "hold",
        "hold",
        "hold",
        "hold",
        "close",
    ]


def test_episode7_saved_rejected_chunk_is_admitted_without_changing_arm():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / (
        "docs/experiments/openpi_recap_franka/pi05_gripper_validation_evidence.json"
    )
    chunk = json.loads(path.read_text())["episode7_rejected_chunk"]["raw_action_7d"]
    adapted = validate_action_chunk(chunk)
    assert [row[:6] for row in adapted] == [row[:6] for row in chunk]
    assert [row[6] for row in adapted[:3]] == [0.0, 0.0, chunk[2][6]]
    assert all(0.0 <= row[6] <= 1.0 for row in adapted)


def test_mapping_verification_preserves_operator_attestation(fr3_mapping_verification):
    import hashlib

    from toolkits.realworld_check.fr3_pi05_request_adapter import (
        load_mapping_verification,
    )

    path, record = fr3_mapping_verification
    result = load_mapping_verification(path)
    assert result == {
        **record,
        "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize(
    "missing",
    [
        f"{axis}_{direction}"
        for axis in ARM_AXES
        for direction in ("positive", "negative")
    ]
    + [
        "controller_acceptance",
        "tool_frame_alignment",
        "timing_10hz_100hz",
        "neutral_stop",
        "gripper_open_close",
    ],
)
def test_identity_gate_requires_every_physical_check(fr3_mapping_verification, missing):
    import json

    from toolkits.realworld_check.fr3_pi05_request_adapter import (
        load_mapping_verification,
    )

    path, record = fr3_mapping_verification
    del record["checks"][missing]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=missing):
        load_mapping_verification(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mapping_mode", "legacy_limited"),
        ("arm_axes", list(reversed(ARM_AXES))),
        ("policy_period_ns", 10_000_000),
        ("publish_period_ns", 100_000_000),
        ("operator", ""),
        ("base_frame", ""),
        ("tool_frame", ""),
        ("controller_binary_sha256", "unverified"),
        ("controller_config_sha256", ""),
        ("verified_at_utc", "2999-01-01T00:00:00Z"),
        ("verified_at_utc", "2020-01-01"),
    ],
)
def test_identity_gate_rejects_wrong_or_incomplete_contract(
    fr3_mapping_verification, field, value
):
    import json

    from toolkits.realworld_check.fr3_pi05_request_adapter import (
        load_mapping_verification,
    )

    path, record = fr3_mapping_verification
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load_mapping_verification(path)


def test_identity_gate_rejects_changed_evidence(fr3_mapping_verification):
    from toolkits.realworld_check.fr3_pi05_request_adapter import (
        load_mapping_verification,
    )

    path, record = fr3_mapping_verification
    evidence = path.parent / record["checks"]["neutral_stop"]["evidence_path"]
    evidence.write_text("Changed evidence\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_mapping_verification(path)


def test_identity_gate_requires_record():
    from toolkits.realworld_check.fr3_pi05_request_adapter import (
        load_mapping_verification,
    )

    with pytest.raises(ValueError, match="requires a physical"):
        load_mapping_verification(None)


@pytest.mark.parametrize("fault", ["missing", "empty", "not_passed", "numeric_passed"])
def test_identity_gate_rejects_unusable_physical_evidence(
    fr3_mapping_verification, fault
):
    import json

    from toolkits.realworld_check.fr3_pi05_request_adapter import (
        load_mapping_verification,
    )

    path, record = fr3_mapping_verification
    check = record["checks"]["neutral_stop"]
    evidence = path.parent / check["evidence_path"]
    if fault == "missing":
        check["evidence_path"] = "absent.txt"
    elif fault == "empty":
        evidence.write_bytes(b"")
    else:
        check["passed"] = False if fault == "not_passed" else 1
    path.write_text(json.dumps(record))
    with pytest.raises((ValueError, OSError)):
        load_mapping_verification(path)
