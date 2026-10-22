# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only focused tests for image/state swap action comparisons."""

import numpy as np
import pytest

from scripts import diagnose_stackcube_pi05_visual_sensitivity as diag


def test_make_input_keeps_same_rgb_state_and_task_contract():
    image = np.full((224, 224, 3), 127, dtype=np.uint8)
    state = np.linspace(0.0, 0.8, 9, dtype=np.float32)
    obs = diag.make_input(image, state)
    assert obs["main_images"].shape == (1, 224, 224, 3)
    assert obs["states"].shape == (1, 9)
    assert obs["task_descriptions"] == [
        "stack the red cube on top of the green cube and release it"
    ]
    np.testing.assert_array_equal(obs["main_images"][0], image)
    np.testing.assert_array_equal(obs["states"][0], state)


def test_action_comparison_separates_arm_radians_and_gripper():
    first = np.zeros((10, 8), dtype=np.float32)
    second = np.zeros_like(first)
    second[:, :7] = 0.2
    second[:, 7] = -0.5
    result = diag.compare_actions(first, second)
    assert result["arm_mae_first_action_rad"] == pytest.approx(0.2)
    assert result["arm_mae_all_10_actions_rad"] == pytest.approx(0.2)
    assert result["gripper_abs_first_action"] == pytest.approx(0.5)
    assert result["gripper_mae_all_10_actions"] == pytest.approx(0.5)


def test_infer_controls_seed_and_validates_action_chunk():
    import torch

    class FakeModel:
        def predict_action_batch(self, obs, mode):
            assert mode == "eval"
            assert obs["main_images"].shape == (1, 224, 224, 3)
            assert obs["states"].shape == (1, 9)
            return torch.randn(1, 10, 8), {}

    image = np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.zeros(9, dtype=np.float32)
    first = diag.infer(FakeModel(), image, state, seed=42)
    repeated = diag.infer(FakeModel(), image, state, seed=42)
    np.testing.assert_array_equal(first, repeated)
    assert first.shape == (10, 8)
