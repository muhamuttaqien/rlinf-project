"""Small offline SFT handoff checks (no OpenPI checkpoint or hardware required)."""

import json

import pytest

from toolkits.realworld_check import fr3_sft_handoff_preflight as p
from toolkits.realworld_check.fr3_labeled_episode_export import ACTION_SOURCE


def _fixture(tmp_path):
    ds = tmp_path / "dataset"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text(json.dumps({
        "features": {
            "state": {"shape": [19]},
            "actions": {"shape": [7]},
            "image": {"shape": [64, 64, 3]},
            "extra_view_image": {"shape": [64, 64, 3]},
        },
        "total_frames": 10, "total_episodes": 1,
    }))
    (ds / "fr3_label_provenance.json").write_text(json.dumps({
        "label_source": ACTION_SOURCE,
        "physical_policy_execution_approved": False,
    }))
    norm = tmp_path / "fr3" / "norm_stats.json"
    norm.parent.mkdir()
    norm.write_text(json.dumps({
        "state": {"mean": [0] * 32}, "actions": {"mean": [0] * 32}
    }))
    return ds, norm


def test_labeled_schema_and_32d_stats(monkeypatch, tmp_path):
    ds, norm = _fixture(tmp_path)
    assert p.inspect_metadata(ds, norm)["fr3_norm_dimensions"] == {
        "state": 32, "actions": 32
    }


def test_simulation_18d_stats_rejected(tmp_path):
    ds, norm = _fixture(tmp_path)
    norm.write_text(json.dumps({
        "state": {"mean": [0] * 18}, "actions": {"mean": [0] * 7}
    }))
    with pytest.raises(ValueError, match="32"):
        p.inspect_metadata(ds, norm)


def test_missing_wrist_rejected(tmp_path):
    ds, norm = _fixture(tmp_path)
    info = json.loads((ds / "meta" / "info.json").read_text())
    del info["features"]["extra_view_image"]
    (ds / "meta" / "info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match="camera"):
        p.inspect_metadata(ds, norm)


def test_missing_action_provenance_rejected(tmp_path):
    ds, norm = _fixture(tmp_path)
    (ds / "fr3_label_provenance.json").write_text("{}")
    with pytest.raises(ValueError, match="provenance"):
        p.inspect_metadata(ds, norm)
