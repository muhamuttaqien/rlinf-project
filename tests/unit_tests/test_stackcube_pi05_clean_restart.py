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

"""Focused tests for clean-v1 configuration, image and policy I/O boundaries."""

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi.modules.model import (
    Observation,
    preprocess_observation,
)
from scripts.evaluate_stackcube_pi05_ee_delta import predict_policy_actions


def test_clean_config_disables_training_only_image_crop():
    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(
        root / "examples/sft/config/"
        "maniskill_stackcube_panda_sft_openpi_pi05_clean_v1.yaml"
    )
    assert cfg.actor.model.openpi.train_image_augmentation is False
    assert cfg.actor.model.openpi.discrete_state_input is True
    assert cfg.actor.model.openpi.num_images_in_input == 2
    assert cfg.actor.model.action_dim == 7
    assert cfg.actor.model.openpi.action_horizon == 10
    assert cfg.runner.logger.experiment_name == "maniskill_stackcube_pi05_clean_v1"


def test_clean_images_preserved_by_nonaugmenting_preprocessing():
    images = {
        "base_0_rgb": torch.linspace(-1.0, 1.0, 224 * 224 * 3).reshape(1, 224, 224, 3),
        "left_wrist_0_rgb": torch.zeros(1, 224, 224, 3),
        "right_wrist_0_rgb": torch.zeros(1, 224, 224, 3),
    }
    obs = Observation(
        images=images,
        image_masks={key: torch.tensor([key != "right_wrist_0_rgb"]) for key in images},
        state=torch.zeros(1, 32),
    )
    prepared = preprocess_observation(obs, train=False)
    for key in images:
        torch.testing.assert_close(prepared.images[key], images[key], rtol=0, atol=0)


def test_policy_inference_accepts_policies_without_rng_kwarg():
    class Model:
        def predict_action_batch(self, obs, mode):
            assert mode == "eval"
            assert obs["states"].shape == (1, 8)
            return np.zeros((1, 10, 7), dtype=np.float32), {}

    sample = {
        "image": np.zeros((224, 224, 3), dtype=np.uint8),
        "wrist_image": np.ones((224, 224, 3), dtype=np.uint8),
        "state": np.zeros(8, dtype=np.float32),
    }
    out = predict_policy_actions(Model(), sample)
    assert out.shape == (1, 10, 7)
