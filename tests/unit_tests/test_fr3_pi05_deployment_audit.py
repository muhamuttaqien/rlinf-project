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
    stats = {"norm_stats": {"actions": {
        "mean": [0.0] * 32,
        "q01": [-1.0] * 7 + [0.0] * 25,
        "q99": [1.0] * 7 + [1.0] * 25,
    }}}
    (tmp_path / "norm_stats.json").write_text(json.dumps(stats))
    report = audit.audit(tmp_path, tmp_path, tmp_path)
    assert report["train_only_norm_passed"]
    assert report["decoded_action_7d_contract"][0] == "linear_x"
    assert report["existing_live_extra_arm_gain"] == 0.15
    assert report["proposed_request_mapping_gain"] == 1.0
    assert report["physical_identity_execution_approved"] is False


def test_refuses_bad_training_action_ranges(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "validate_metadata", lambda *_args: {
        "train": {"source_episodes": 80},
        "validation": {"source_episodes": 20},
        "train_only_norm_passed": True,
        "source_episode_disjoint": True,
    })
    stats = {"norm_stats": {"actions": {
        "mean": [0.0] * 32, "q01": [-1.0] * 32,
        "q99": [1.4] + [1.0] * 31,
    }}}
    (tmp_path / "norm_stats.json").write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="normalized SpaceMouse"):
        audit.audit(tmp_path, tmp_path, tmp_path)
