# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Read-only deployment provenance audit with synthetic metadata; no hardware."""

import json

import pytest

from toolkits.realworld_check import fr3_pi05_deployment_audit as audit


def test_request_audit_reports_unverified_physical_mapping(tmp_path, monkeypatch):
    def provenance(*_args):
        return {
            "train": {"source_episodes": 80},
            "validation": {"source_episodes": 20},
            "train_only_norm_passed": True,
            "source_episode_disjoint": True,
        }

    monkeypatch.setattr(audit, "validate_metadata", provenance)
    stats = {
        "norm_stats": {
            "actions": {
                "mean": [0.0] * 32,
                "q01": [-1.0] * 7 + [0.0] * 25,
                "q99": [1.0] * 7 + [1.0] * 25,
            }
        }
    }
    (tmp_path / "norm_stats.json").write_text(json.dumps(stats))
    report = audit.audit(tmp_path, tmp_path, tmp_path)
    assert report["train_only_norm_passed"]
    assert report["decoded_action_7d_contract"][0] == "linear_x"
    assert report["existing_live_extra_arm_gain"] == 0.15
    assert report["proposed_request_mapping_gain"] == 1.0
    assert report["physical_identity_execution_approved"] is False


def test_refuses_bad_training_action_ranges(tmp_path, monkeypatch):
    monkeypatch.setattr(
        audit,
        "validate_metadata",
        lambda *_args: {
            "train": {"source_episodes": 80},
            "validation": {"source_episodes": 20},
            "train_only_norm_passed": True,
            "source_episode_disjoint": True,
        },
    )
    stats = {
        "norm_stats": {
            "actions": {
                "mean": [0.0] * 32,
                "q01": [-1.0] * 32,
                "q99": [1.4] + [1.0] * 31,
            }
        }
    }
    (tmp_path / "norm_stats.json").write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="normalized SpaceMouse"):
        audit.audit(tmp_path, tmp_path, tmp_path)


@pytest.fixture
def recorded_requests(tmp_path):
    raw = tmp_path / "raw" / "episode_0001"
    raw.mkdir(parents=True)
    commands = []
    row = [0.5, -0.25, 0.75, -0.1, 0.2, -0.3]
    for i in range(3):
        commands.append(
            {
                "name": "operator_twist",
                "topic": "/franka_controller/target_cartesian_velocity_percent",
                "message_type": "geometry_msgs/msg/Twist",
                "units": {
                    "linear": "normalized_controller_input_not_m_per_s",
                    "angular": "normalized_controller_input_not_rad_per_s",
                },
                "message": {
                    part: dict(zip(("x", "y", "z"), row[offset : offset + 3]))
                    for part, offset in (("linear", 0), ("angular", 3))
                },
                "received_monotonic_ns": 2_000_000_000 + i * 10_000_000,
            }
        )
    for value in (0.0, 1.0):
        commands.append(
            {
                "name": "operator_gripper_request",
                "topic": "/gripper_client/target_gripper_width_percent",
                "message_type": "std_msgs/msg/Float32",
                "units": {"data": "normalized_open_fraction"},
                "message": {"data": value},
            }
        )
    (raw / "operator_commands.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in commands)
    )
    episode = tmp_path / "policy"
    episode.mkdir()
    events = [
        {
            "event": "action",
            "command_index": i % 10,
            "requested_7d": row + [1.0],
            "sent_arm_6d": [v * 0.15 for v in row],
            "monotonic_ns": 3_000_000_000
            + (i // 10) * 1_484_000_000
            + (i % 10) * 100_000_000,
            "gripper_event": "open" if i == 2 else "hold",
        }
        for i in range(20)
    ]
    (episode / "policy_command_log.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    return raw.parent, episode


def test_recorded_comparison_preserves_units_axes_and_two_mapping_modes(
    recorded_requests,
):
    raw, episode = recorded_requests
    report = audit.compare_recorded_requests(raw, episode, expected_episodes=1)
    assert report["demonstrations"] == 1
    assert report["raw_requests"] == 3
    assert report["raw_effective_hz"] == 100.0
    assert report["raw_receipt_gaps"]["maximum_ms"] == 10.0
    assert report["policy_within_chunk_gaps"]["maximum_ms"] == 100.0
    assert report["policy_between_chunk_gaps"]["maximum_ms"] == 584.0
    assert report["legacy_mapping_max_absolute_error"] == 0.0
    assert report["identity_mapping_max_absolute_error"] == 0.0
    assert report["identity_sent_distribution"] == report["decoded_distribution"]
    assert report["raw_gripper_requests"] == {"0.0": 1, "1.0": 1}
    assert report["offline_debounced_gripper_events"] == {"hold": 19, "open": 1}
    assert report["physical_identity_execution_approved"] is False


@pytest.mark.parametrize(
    "fault", ["units", "out_of_range", "gripper", "remapping", "timestamp"]
)
def test_recorded_comparison_rejects_contract_drift(recorded_requests, fault):
    raw, episode = recorded_requests
    path = (
        episode / "policy_command_log.jsonl"
        if fault in ("remapping", "timestamp")
        else raw / "episode_0001" / "operator_commands.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if fault == "units":
        rows[0]["units"]["linear"] = "m_per_s"
    elif fault == "out_of_range":
        rows[0]["message"]["linear"]["x"] = 1.001
    elif fault == "gripper":
        rows[-1]["message"]["data"] = 0.5
    elif fault == "remapping":
        rows[0]["sent_arm_6d"][0] *= -1
    else:
        rows[1]["monotonic_ns"] = rows[0]["monotonic_ns"]
    path.write_text("".join(json.dumps(e) + "\n" for e in rows))
    with pytest.raises(ValueError):
        audit.compare_recorded_requests(raw, episode, expected_episodes=1)


def test_recorded_comparison_requires_all_expected_demonstrations(recorded_requests):
    raw, episode = recorded_requests
    with pytest.raises(ValueError, match="Expected 100"):
        audit.compare_recorded_requests(raw, episode)
