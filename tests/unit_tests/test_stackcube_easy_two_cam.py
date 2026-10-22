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

"""Focused easy StackCube two-camera contract tests; no GPU needed."""

import numpy as np
import pytest
from omegaconf import OmegaConf

from scripts import evaluate_stackcube_pi05 as evaluator
from toolkits.lerobot.collect_maniskill_stackcube_recap import (
    _extract_observation,
)
from rlinf.data.datasets.openpi.v3_stackcube_dataset import V3StackCubeDataset


def rgb_obs(*, hand: bool) -> dict:
    image = np.full((1, 224, 224, 3), 40, dtype=np.uint8)
    sensors = {"base_camera": {"rgb": image}}
    if hand:
        sensors["hand_camera"] = {
            "rgb": np.full((1, 224, 224, 3), 210, dtype=np.uint8)
        }
    return {
        "agent": {"qpos": np.zeros((1, 9), dtype=np.float32)},
        "sensor_data": sensors,
    }


def test_two_cameras_are_distinct_and_missing_wrist_is_rejected():
    parsed = _extract_observation(rgb_obs(hand=True), require_wrist_camera=True)
    assert parsed["image"].shape == (224, 224, 3)
    assert parsed["wrist_image"].shape == (224, 224, 3)
    assert not np.array_equal(parsed["image"], parsed["wrist_image"])

    model_obs, _ = evaluator.extract_stackcube_obs(
        rgb_obs(hand=True), use_wrist_camera=True
    )
    assert model_obs["main_images"].shape == (1, 224, 224, 3)
    assert model_obs["wrist_images"].shape == (1, 224, 224, 3)
    assert model_obs["states"].shape == (1, 9)
    with pytest.raises(ValueError, match="real hand_camera"):
        _extract_observation(rgb_obs(hand=False), require_wrist_camera=True)
    with pytest.raises(ValueError, match="real hand_camera"):
        evaluator.extract_stackcube_obs(rgb_obs(hand=False), use_wrist_camera=True)


def test_v3_adapter_uses_real_wrist_only_when_enabled(monkeypatch):
    """Avoid actual Parquet I/O to check the selected output keys and masks."""
    adapter = V3StackCubeDataset.__new__(V3StackCubeDataset)
    adapter.frames = [{"actions": [0.0] * 8, "state": [0.0] * 9,
                       "episode_index": 0, "frame_index": 0}]
    adapter._last_index = np.array([0], dtype=np.int64)
    adapter.action_horizon = 10
    adapter.state_dim = 9
    adapter.action_dim = 8

    def fake_image(frame, *, image_key="image"):
        return np.full(
            (224, 224, 3), 10 if image_key == "image" else 230, dtype=np.uint8
        )

    monkeypatch.setattr(adapter, "_embedded_image", fake_image)
    adapter.use_wrist_image = True
    pair = adapter[0]
    assert pair["wrist_image"][0, 0, 0] == 230
    assert pair["image"][0, 0, 0] == 10
    assert pair["actions"].shape == (10, 8)

    adapter.use_wrist_image = False
    single = adapter[0]
    assert np.count_nonzero(single["wrist_image"]) == 0


def test_new_config_is_separate_from_previous_baseline():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    old_yaml = OmegaConf.load(
        root / "examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_state.yaml"
    )
    new_yaml = OmegaConf.load(
        root / "examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_easy_two_cam.yaml"
    )
    assert old_yaml.actor.model.openpi.num_images_in_input == 1
    assert new_yaml.actor.model.openpi.num_images_in_input == 2
    assert new_yaml.actor.model.openpi.discrete_state_input is True
    assert new_yaml.actor.model.openpi.config_name == (
        "pi05_stackcube_panda_easy_two_cam_state"
    )
    assert new_yaml.runner.max_steps == 600
    assert new_yaml.actor.model.action_dim == 8
    assert new_yaml.actor.model.openpi.action_horizon == 10


def test_easy_spawn_region_narrow_and_collision_free():
    pytest.importorskip("mani_skill")
    import torch
    from toolkits.lerobot.easy_stackcube_env import (
        GREEN_CENTER_XY, RED_CENTER_XY, XY_HALF_WIDTH_M,
        sample_cube_poses,
    )

    torch.manual_seed(123)
    pos, quat = sample_cube_poses(100, "cpu")
    assert pos.shape == (2, 100, 3)
    assert quat.shape == (2, 100, 4)
    assert torch.allclose(pos[:, :, 2], torch.full((2, 100), 0.02))
    for i, center in enumerate((RED_CENTER_XY, GREEN_CENTER_XY)):
        error = torch.abs(pos[i, :, :2] - torch.tensor(center))
        assert torch.all(error <= XY_HALF_WIDTH_M + 1e-7)
    distance = torch.linalg.norm(pos[0, :, :2] - pos[1, :, :2], dim=-1)
    assert float(distance.min()) > 0.10



def test_registered_openpi_two_camera_config_enables_real_wrist_transform():
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.policies.maniskill_policy import ManiSkillInputs

    config = get_openpi_config("pi05_stackcube_panda_easy_two_cam_state")
    assert config.model.discrete_state_input is True
    assert config.model.action_horizon == 10
    assert config.data.use_wrist_image is True
    data = config.data.create(config.assets_dirs, config.model)
    inputs = [
        transform for transform in data.data_transforms.inputs
        if isinstance(transform, ManiSkillInputs)
    ]
    assert len(inputs) == 1 and inputs[0].use_wrist_image is True
    observation = {
        "observation/image": np.full((224, 224, 3), 10, dtype=np.uint8),
        "observation/wrist_image": np.full((224, 224, 3), 210, dtype=np.uint8),
        "observation/state": np.zeros(9, dtype=np.float32),
    }
    transformed = inputs[0](observation)
    assert bool(transformed["image_mask"]["base_0_rgb"])
    assert bool(transformed["image_mask"]["left_wrist_0_rgb"])
    assert not bool(transformed["image_mask"]["right_wrist_0_rgb"])
    assert not np.array_equal(
        transformed["image"]["base_0_rgb"],
        transformed["image"]["left_wrist_0_rgb"],
    )
