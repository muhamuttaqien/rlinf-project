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


def test_cfg_state_dict_requires_matching_keys():
    from rlinf.models.embodiment.openpi_cfg import (
        _prepare_cfg_checkpoint_state_dict,
    )

    model = torch.nn.Linear(3, 2)
    valid = model.state_dict()
    assert _prepare_cfg_checkpoint_state_dict(model, valid, "valid.pt") == valid
    with pytest.raises(RuntimeError, match="incompatible parameters"):
        _prepare_cfg_checkpoint_state_dict(model, {"img.stem.weight": torch.zeros(2)}, "invalid.pt")


def test_cfg_state_dict_permits_only_unused_expert_lm_head_missing():
    from rlinf.models.embodiment.openpi_cfg import (
        _prepare_cfg_checkpoint_state_dict,
        _UNUSED_EXPERT_LM_HEAD,
    )

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.action_in_proj = torch.nn.Linear(2, 2)
            self.register_parameter(
                "dummy_unused", torch.nn.Parameter(torch.zeros(1))
            )

    model = ToyModel()
    # Missing a required key must never be silently ignored.
    incomplete = {"action_in_proj.weight": model.action_in_proj.weight.detach()}
    with pytest.raises(RuntimeError, match="incompatible parameters"):
        _prepare_cfg_checkpoint_state_dict(model, incomplete, "incomplete.pt")


def test_rlinf_token_embedding_maps_to_paligemma_input_and_head():
    from rlinf.utils.ckpt_convertor.openpi.openpi_to_openpi_pytorch import (
        new_to_old_state_dict,
    )

    weight = torch.ones(3, 4)
    converted = new_to_old_state_dict({"llm.embedder.embedding.weight": weight})
    assert torch.equal(
        converted[
            "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        ],
        weight,
    )
    assert torch.equal(
        converted["paligemma_with_expert.paligemma.lm_head.weight"],
        weight,
    )


def test_cfg_restore_tied_embedding_alias():
    from rlinf.models.embodiment.openpi_cfg import (
        _PALIGEMMA_INPUT_EMBED,
        _PALIGEMMA_LM_HEAD,
        _restore_tied_paligemma_weights,
    )

    shared = torch.ones(5, 3)
    expected = {_PALIGEMMA_INPUT_EMBED: shared, _PALIGEMMA_LM_HEAD: shared}
    saved = {_PALIGEMMA_LM_HEAD: shared.clone()}
    restored = _restore_tied_paligemma_weights(expected, saved, "sft.safetensors")
    assert set(restored) == set(expected)
    assert torch.equal(restored[_PALIGEMMA_INPUT_EMBED], saved[_PALIGEMMA_LM_HEAD])


def test_cfg_refuse_tied_embedding_alias_if_model_weights_are_not_tied():
    from rlinf.models.embodiment.openpi_cfg import (
        _PALIGEMMA_INPUT_EMBED,
        _PALIGEMMA_LM_HEAD,
        _restore_tied_paligemma_weights,
    )

    expected = {
        _PALIGEMMA_INPUT_EMBED: torch.ones(5, 3),
        _PALIGEMMA_LM_HEAD: torch.ones(5, 3),
    }
    with pytest.raises(RuntimeError, match="does not share"):
        _restore_tied_paligemma_weights(
            expected, {_PALIGEMMA_LM_HEAD: torch.ones(5, 3)}, "invalid.safetensors"
        )
