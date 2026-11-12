"""Small offline tests: labels are never inferred from raw FR3 intent."""

import hashlib
import json

import numpy as np
import pytest

from toolkits.realworld_check import fr3_labeled_episode_export as exp
from toolkits.realworld_check.fr3_policy_contract import ACTION_FIELDS, CONTRACT_VERSION


def _fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = {
        "task": "pick and place", "sample_rate_hz_requested": 10,
        "camera_profiles": {"base": {"resolution": [1, 1]},
                            "wrist": {"resolution": [1, 1]}},
    }
    manifest_path = raw / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    measured = {
        "tcp_pose": [0, 0, 0, 0, 0, 0, 1],
        "arm_joint_position": [0] * 7,
        "tcp_force": [0, 0, 0],
        "tcp_torque": [0, 0, 0],
        "robotiq_knuckle_position": [0.1],
    }
    data = [
        {"recorded_monotonic_ns": 1000000000 + 100000000*i,
         "measured_state": measured}
        for i in range(2)
    ]
    (raw / "states.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in data)
    )
    label = {
        "schema": exp.SCHEMA,
        "raw_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "task": "pick and place",
        "state_contract": CONTRACT_VERSION,
        "action_field_names": list(ACTION_FIELDS),
        "action_source": exp.ACTION_SOURCE,
        "episode_success": True, "human_reviewed": True,
        "reviewer": "test reviewer", "controller_telemetry_source": "test fixture",
        "verification": {key: True for key in exp.REVIEW_KEYS},
        "clock": {"nuc_minus_workstation_monotonic_ns": 5000000000,
                  "uncertainty_ns": 100000,
                  "max_skew_ns": 2000000},
        "records": [
            {"frame_index": i,
             "controller_accepted_ns": 6000100000 + 100000000*i,
             "source_record_id": "controller-%d" % i,
             "action": [0.001, 0, 0, 0, 0, 0, 0.2]}
            for i in range(2)
        ],
    }
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(label))
    return raw, p, label


def test_valid_external_target_labels_are_not_execution_approval(monkeypatch, tmp_path):
    raw, p, _ = _fixture(tmp_path)
    monkeypatch.setattr(exp, "audit_raw_episode",
                        lambda _: ({"sample_count": 2}, {}))
    result, frames = exp.check_labels(raw, p)
    assert len(frames) == 2
    assert frames[0]["action"].shape == (7,)
    assert frames[0]["state"].shape == (19,)
    assert result["policy_training_data_exportable"] is True
    assert result["physical_policy_execution_approved"] is False


def test_mismatched_manifest_blocked(monkeypatch, tmp_path):
    raw, p, label = _fixture(tmp_path)
    monkeypatch.setattr(exp, "audit_raw_episode",
                        lambda _: ({"sample_count": 2}, {}))
    label["raw_manifest_sha256"] = "bad"
    p.write_text(json.dumps(label))
    with pytest.raises(ValueError, match="SHA256"):
        exp.check_labels(raw, p)


def test_operator_intent_not_training_target(monkeypatch, tmp_path):
    raw, p, label = _fixture(tmp_path)
    monkeypatch.setattr(exp, "audit_raw_episode",
                        lambda _: ({"sample_count": 2}, {}))
    label["action_source"] = "spacemouse_normalized_twist"
    p.write_text(json.dumps(label))
    with pytest.raises(ValueError, match="SpaceMouse"):
        exp.check_labels(raw, p)


def test_out_of_alignment_blocked(monkeypatch, tmp_path):
    raw, p, label = _fixture(tmp_path)
    monkeypatch.setattr(exp, "audit_raw_episode",
                        lambda _: ({"sample_count": 2}, {}))
    label["records"][1]["controller_accepted_ns"] += 100000000
    p.write_text(json.dumps(label))
    with pytest.raises(ValueError, match="skew"):
        exp.check_labels(raw, p)


def test_incomplete_actions_blocked(monkeypatch, tmp_path):
    raw, p, label = _fixture(tmp_path)
    monkeypatch.setattr(exp, "audit_raw_episode",
                        lambda _: ({"sample_count": 2}, {}))
    label["records"].pop()
    p.write_text(json.dumps(label))
    with pytest.raises(ValueError, match="EVERY"):
        exp.check_labels(raw, p)
