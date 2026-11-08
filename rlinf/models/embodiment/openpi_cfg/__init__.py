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
        _validate_cfg_checkpoint_keys(model, model_state_dict, full_weights_path)
        model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu", weights_only=False)
        _validate_cfg_checkpoint_keys(model, model_state_dict, actor_full_weights_path)
        model.load_state_dict(model_state_dict, strict=False)
    else:
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        for weight_path in weight_paths:
            with safetensors.safe_open(weight_path, framework="pt", device="cpu") as reader:
                _validate_cfg_checkpoint_keys(model, reader.keys(), weight_path)
            safetensors.torch.load_model(model, weight_path, strict=False)

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
