"""CFG checkpoint loading must reject incompatible OpenPI layouts."""
import pytest
import torch

from rlinf.models.embodiment.openpi_cfg import _validate_cfg_checkpoint_keys


def test_cfg_checkpoint_keys_accept_matching_state_dict():
    model = torch.nn.Linear(3, 2)
    _validate_cfg_checkpoint_keys(model, model.state_dict().keys(), "valid.pt")


def test_cfg_checkpoint_keys_reject_unmatched_format():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    with pytest.raises(RuntimeError, match="Incompatible CFG policy checkpoint"):
        _validate_cfg_checkpoint_keys(
            model,
            ["img.stem.weight", "llm.layers.0.attn.q_proj.0.weight"],
            "new_openpi.safetensors",
        )
