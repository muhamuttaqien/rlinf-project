"""Small filesystem tests for offline π0.5 checkpoint path handling."""

import pytest

from toolkits.realworld_check.fr3_pi05_checkpoint_smoke import resolve_checkpoint


def test_step_dir_resolves(tmp_path):
    p = tmp_path / "actor" / "model_state_dict"
    p.mkdir(parents=True)
    (p / "full_weights.pt").touch()
    assert resolve_checkpoint(tmp_path) == p / "full_weights.pt"


def test_full_weights_resolves(tmp_path):
    p = tmp_path / "full_weights.pt"
    p.touch()
    assert resolve_checkpoint(p) == p


def test_missing_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="full_weights"):
        resolve_checkpoint(tmp_path)
