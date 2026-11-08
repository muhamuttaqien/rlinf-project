# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import torch
from omegaconf import DictConfig


def _validate_cfg_checkpoint_keys(model, checkpoint_keys, checkpoint_path):
    """Fail fast when a CFG policy would silently skip most pretrained weights."""
    expected = set(model.state_dict())
    observed = set(checkpoint_keys)
    matching = expected & observed
    coverage = len(matching) / max(len(expected), 1)
    if coverage < 0.95:
        raise RuntimeError(
            f"Incompatible CFG policy checkpoint at {checkpoint_path}: "
            f"only {len(matching)}/{len(expected)} model tensors have matching names "
            f"({coverage:.1%}). The new RLinf Pi0 format (img.*, llm.*) is "
            "not directly loadable into the legacy OpenPI PyTorch CFG model "
            "(paligemma_with_expert.*). Using strict=False would silently "
            "initialize nearly the whole policy randomly. Supply a compatible "
            "OpenPI PyTorch checkpoint; do not train with this checkpoint."
        )


# OpenPI's PyTorch action-expert language head is absent from the RLinf
# inference-focused Pi0 format. CFG predicts actions from hidden states,
# not logits from this language head.
_UNUSED_EXPERT_LM_HEAD = "paligemma_with_expert.gemma_expert.lm_head.weight"

_PALIGEMMA_INPUT_EMBED = (
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
)
_PALIGEMMA_LM_HEAD = "paligemma_with_expert.paligemma.lm_head.weight"


def _restore_tied_paligemma_weights(expected, state_dict, checkpoint_path):
    """Restore a weight alias that safetensors may omit for tied PaliGemma weights."""
    aliases = (_PALIGEMMA_INPUT_EMBED, _PALIGEMMA_LM_HEAD)
    missing = [key for key in aliases if key in expected and key not in state_dict]
    present = [key for key in aliases if key in state_dict]
    if len(missing) == 1 and len(present) == 1:
        missing_key, source_key = missing[0], present[0]
        if (
            source_key not in expected
            or expected[missing_key].shape != state_dict[source_key].shape
            or expected[missing_key].data_ptr() != expected[source_key].data_ptr()
        ):
            raise RuntimeError(
                f"Cannot restore tied PaliGemma weight in {checkpoint_path}: "
                "model does not share the expected embedding/head storage or "
                "tensor shapes do not match."
            )
        state_dict = dict(state_dict)
        state_dict[missing_key] = state_dict[source_key]
        print(
            f"Restored tied PaliGemma weight: {missing_key} <- {source_key}",
            flush=True,
        )
    return state_dict


def _prepare_cfg_checkpoint_state_dict(model, state_dict, checkpoint_path):
    """Convert RLinf Pi0 weights to OpenPI PyTorch CFG layout and validate all used weights."""
    if any(key.startswith(("img.", "llm.")) for key in state_dict):
        from rlinf.utils.ckpt_convertor.openpi.openpi_to_openpi_pytorch import (
            new_to_old_state_dict,
        )

        state_dict = new_to_old_state_dict(
            state_dict, action_expert_uses_adarms=bool(model.config.pi05)
        )
        print(f"Converted RLinf Pi0 checkpoint for CFG: {checkpoint_path}", flush=True)

    expected = model.state_dict()
    state_dict = _restore_tied_paligemma_weights(expected, state_dict, checkpoint_path)
    missing = set(expected) - set(state_dict)
    unexpected = set(state_dict) - set(expected)
    shape_mismatches = [
        key for key in expected.keys() & state_dict.keys()
        if expected[key].shape != state_dict[key].shape
    ]
    if missing - {_UNUSED_EXPERT_LM_HEAD} or unexpected or shape_mismatches:
        raise RuntimeError(
            f"CFG checkpoint {checkpoint_path} has incompatible parameters: "
            f"missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]}, "
            f"shape_mismatches={sorted(shape_mismatches)[:8]}. "
            "Do not train or evaluate until checkpoint conversion matches."
        )
    if missing:
        print("CFG: leaving unused action-expert language head at initialization", flush=True)
    print(f"CFG checkpoint validated: {len(expected) - len(missing)}/{len(expected)} "
          "model tensors matched", flush=True)
    return state_dict


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
        OpenPi0Config,
        OpenPi0ForCFGActionPrediction,
    )

    config_name = getattr(cfg.openpi, "config_name", None)
    actor_train_config = get_openpi_config(config_name)
    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_config_kwargs = cfg.openpi
    if override_config_kwargs is not None:
        for key, val in override_config_kwargs.items():
            actor_model_config.__dict__[key] = val
    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    model: OpenPi0ForCFGActionPrediction = OpenPi0ForCFGActionPrediction(
        actor_model_config
    )
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    if os.path.exists(full_weights_path):
        model_state_dict = torch.load(full_weights_path, map_location="cpu", weights_only=False)
        model_state_dict = _prepare_cfg_checkpoint_state_dict(
            model, model_state_dict, full_weights_path
        )
        model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu", weights_only=False)
        model_state_dict = _prepare_cfg_checkpoint_state_dict(
            model, model_state_dict, actor_full_weights_path
        )
        model.load_state_dict(model_state_dict, strict=False)
    else:
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        for weight_path in weight_paths:
            state_dict = safetensors.torch.load_file(weight_path, device="cpu")
            state_dict = _prepare_cfg_checkpoint_state_dict(model, state_dict, weight_path)
            model.load_state_dict(state_dict, strict=False)

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)

    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
