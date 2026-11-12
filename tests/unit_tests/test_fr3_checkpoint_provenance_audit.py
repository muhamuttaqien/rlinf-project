"""Small actual-file tests for read-only checkpoint provenance inspection."""

import json

import pytest

from toolkits.realworld_check.fr3_checkpoint_provenance_audit import (
    build_report, norm_dimensions, scan_experiment_metadata,
)


def test_norm_dim_mismatch_never_authorizes(tmp_path):
    p = tmp_path / "norm_stats.json"
    p.write_text(json.dumps({
        "norm_stats": {
            "state": {"mean": [0] * 18, "std": [1] * 18},
            "actions": {"q01": [0] * 7, "q99": [1] * 7},
        },
    }))
    result = norm_dimensions(p)
    assert result["state_dim"] == 18
    assert result["action_dim"] == 7
    assert result["state_matches_candidate_FR3"] is False
    assert result["action_width_matches_candidate_FR3"] is True
    assert result["approved_for_FR3"] is False
    assert result["sim_stats_applied"] is False


def test_malformed_stat_lengths_are_rejected(tmp_path):
    p = tmp_path / "norm_stats.json"
    p.write_text(json.dumps({
        "state": {"mean": [0, 0], "std": [1]},
    }))
    with pytest.raises(ValueError, match="inconsistent"):
        norm_dimensions(p)


def test_actual_experiment_metadata_found_not_approved(tmp_path):
    ckpt = tmp_path / "full_weights.pt"
    ckpt.write_bytes(b"fake_header_no_torch_load")
    norm = tmp_path / "norm_stats.json"
    norm.write_text(json.dumps({
        "state": {"mean": [0] * 18},
        "actions": {"mean": [0] * 7},
    }))
    exp = tmp_path / "run"
    hydra = exp / ".hydra"
    hydra.mkdir(parents=True)
    (hydra / "config.yaml").write_text(
        "config_name: pi05_maniskill\n"
        "action_dim: 7\n"
        "repo_id: local_stackcube\n"
    )
    report = build_report(ckpt, norm, exp)
    files = report["experiment_metadata"]["files"]
    assert len(files) == 1
    assert files[0]["relative_path"] == ".hydra/config.yaml"
    assert len(files[0]["matched_lines"]) == 3
    assert report["model_action_semantics_verified"] is False
    assert report["FR3_normalization_ready"] is False
    assert report["robot_commands_sent"] is False


def test_missing_experiment_is_explicit(tmp_path):
    result = scan_experiment_metadata(tmp_path / "does-not-exist")
    assert result["exists"] is False
    assert result["files"] == []
