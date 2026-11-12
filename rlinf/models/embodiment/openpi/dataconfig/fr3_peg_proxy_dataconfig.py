# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Experimental OFFLINE Pi0.5 data config for observed FR3 peg trajectory proxies.

These seven values are inferred from *future measured states*, NOT controller
commands. Never use this policy config for live robot execution.
"""

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import fr3_peg_proxy_policy


@dataclasses.dataclass(frozen=True)
class Fr3PegProxyDataConfig(DataConfigFactory):
    default_prompt: str = "Insert the peg into the hole."

    @override
    def create(self, assets_dirs: pathlib.Path,
               model_config: _model.BaseModelConfig) -> DataConfig:
        repack = _transforms.Group(inputs=[
            _transforms.RepackTransform({
                "observation/image": "image",
                "observation/wrist_image": "extra_view_image",
                "observation/state": "state",
                "actions": "actions",
                "prompt": "prompt",
            }),
        ])
        data = _transforms.Group(
            inputs=[fr3_peg_proxy_policy.Fr3PegProxyInputs(
                action_dim=model_config.action_dim,
                model_type=model_config.model_type,
            )],
            outputs=[fr3_peg_proxy_policy.Fr3PegProxyOutputs()],
        )
        model = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data,
            model_transforms=model,
            action_sequence_keys=("actions",),
        )
