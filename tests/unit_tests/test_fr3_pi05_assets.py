"""Focused, hardware-free offline asset provenance tests."""

import json
import pytest

from toolkits.realworld_check.fr3_pi05_checkpoint_smoke import (
    official_tokenizer_cache_path, tokenize_prompt_offline, audit_sim_norm_stats,
)


def test_mismatch_not_approved(tmp_path):
    p = tmp_path / "norm_stats.json"
    p.write_text(json.dumps({"norm_stats": {
        "state": {"mean": [0.0] * 9},
        "actions": {"mean": [0.0] * 8},
    }}))
    result = audit_sim_norm_stats(p)
    assert result["observed_dimensions"] == {"state": 9, "actions": 8}
    assert result["dimension_match"] is False
    assert result["approved_for_FR3"] is False
    assert result["applied_to_inference"] is False


def test_same_shape_still_not_approved(tmp_path):
    p = tmp_path / "norm_stats.json"
    p.write_text(json.dumps({
        "state": {"mean": [0.0] * 19},
        "actions": {"mean": [0.0] * 7},
    }))
    result = audit_sim_norm_stats(p)
    assert result["dimension_match"] is True
    assert result["approved_for_FR3"] is False


def test_missing_official_tokenizer_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path))
    assert official_tokenizer_cache_path() == (
        tmp_path / "big_vision" / "paligemma_tokenizer.model"
    )
    with pytest.raises(FileNotFoundError, match="PaliGemma"):
        tokenize_prompt_offline("pick up the object")
