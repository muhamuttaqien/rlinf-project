# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0
"""LeRobot RGB/7D end-effector adapter for ManiSkill Panda StackCube-v1."""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import maniskill_policy


@dataclasses.dataclass(frozen=True)
class LeRobotStackCubePandaDataConfig(DataConfigFactory):
    """Map converted StackCube observations to RLinf's existing ManiSkill policy."""

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        # LeRobot stores single actions under "action". The official OpenPI
        # loader assembles future action windows before repacking.
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.base_camera",
                        "observation/wrist_image": "observation.images.hand_camera",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "task": "task",
                    }
                )
            ]
        )
        # ManiSkill pd_ee_delta_pose actions are already delta-pose commands.
        # Do NOT apply DeltaActions to them a second time.
        data_transforms = _transforms.Group(
            inputs=[
                maniskill_policy.ManiSkillInputs(
                    model_type=model_config.model_type,
                    use_wrist_image=True,
                )
            ],
            outputs=[maniskill_policy.ManiSkillOutputs(output_action_dim=7)],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
        )
