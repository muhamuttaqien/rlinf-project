"""Synthetic-only Milestone 4C.2T validation (never ROS or hardware)."""

from copy import deepcopy

import pytest

from toolkits.realworld_check.fr3_target_telemetry_contract import (
    SCHEMA, audit_telemetry,
)


def tick():
    q = [0.0] * 7
    target = [0.2] * 7
    return {
        "schema": SCHEMA, "event": "arm_tick", "clock_id": "nuc_monotonic_ns",
        "seq": 0, "tick_ns": 1_000_000_000, "activation_ns": 900_000_000,
        "intent_recv_ns": 999_900_000, "sample_ns": 999_900_000,
        "target_state_ns": 999_900_000, "target_command_ns": 999_900_000,
        "target_activation_ns": 900_000_000, "ik_solve_ns": 999_950_000,
        "intent_axes": [0.5, 0, 0, 0, 0, 0],
        "ik_target_q_rad": target, "reference_before_rad": q,
        "reference_after_rad": [0.001] * 7, "measured_q_rad": q,
        "velocity_limits_rad_s": [1.0] * 7, "dt_s": 0.001,
        "command_timeout_s": 0.25,
        "gates": {"moving": True, "intent_fresh": True, "target_valid": True,
                  "activation_matches": True, "state_fresh": True,
                  "command_fresh": True},
        "reference_advanced": True, "telemetry_drops_total": 0,
    }


def gripper():
    return {
        "schema": SCHEMA, "event": "gripper_goal",
        "clock_id": "nuc_monotonic_ns", "seq": 0, "goal_id": "synthetic-goal-001",
        "request_ns": 1_000_000_100, "sent_ns": 1_000_000_200,
        "accepted_ns": 1_000_000_300, "result_ns": 1_000_000_400,
        "normalized_request": 0.0, "goal_target_knuckle_rad": 0.7929,
        "max_effort": 50.0, "accepted": True, "result_status": 4,
        "measured_knuckle_rad": 0.7859, "measured_ns": 1_000_000_500,
        "replaced_requests_since_last_goal": 0,
    }


def test_synthetic_tick_and_gripper_report_is_not_training_ready():
    r = audit_telemetry([tick(), gripper()])
    assert r["record_count"] == 2
    assert r["arm_reference_advanced_count"] == 1
    assert r["gripper_goal_count"] == 1
    assert r["7d_training_actions_available"] is False
    assert r["cross_host_clock_alignment_verified"] is False


def test_hold_if_neutral():
    r = tick()
    r["intent_axes"] = [0.0] * 6
    r["gates"]["moving"] = False
    r["reference_advanced"] = False
    r["reference_after_rad"] = r["reference_before_rad"][:]
    assert audit_telemetry([r])["arm_reference_held_count"] == 1


def test_stale_command_holds_reference():
    r = tick()
    r["target_command_ns"] = 700_000_000
    r["gates"]["command_fresh"] = False
    r["reference_advanced"] = False
    r["reference_after_rad"] = r["reference_before_rad"][:]
    audit_telemetry([r])


@pytest.mark.parametrize("change", [
    lambda r: r["reference_after_rad"].__setitem__(0, 0.05),
    lambda r: r["gates"].__setitem__("intent_fresh", False),
    lambda r: r["intent_axes"].__setitem__(0, 3.0),
    lambda r: r.update({"target_activation_ns": 1}),
    lambda r: r.update({"ik_target_q_rad": [float("nan")] * 7}),
])
def test_invalid_reference_freshness_and_axes_rejected(change):
    r = tick()
    change(r)
    with pytest.raises(ValueError):
        audit_telemetry([r])


def test_invalid_gripper_mapping_and_lifecycle_rejected():
    a = gripper()
    a["goal_target_knuckle_rad"] = 0.08
    with pytest.raises(ValueError, match="mapping"):
        audit_telemetry([a])
    b = gripper()
    b["accepted_ns"] = None
    with pytest.raises(ValueError):
        audit_telemetry([b])


def test_sequence_gaps_counted_and_repeats_rejected():
    a = tick()
    b = deepcopy(a)
    b["seq"] = 3
    b["tick_ns"] += 1_000_000
    assert audit_telemetry([a, b])["sequence_gaps"] == 2
    b["seq"] = 0
    with pytest.raises(ValueError, match="sequence"):
        audit_telemetry([a, b])


def test_no_event_or_unknown_schema():
    with pytest.raises(ValueError, match="No telemetry"):
        audit_telemetry([])
    r = tick()
    r["schema"] = "raw_capture_v1"
    with pytest.raises(ValueError, match="schema"):
        audit_telemetry([r])
