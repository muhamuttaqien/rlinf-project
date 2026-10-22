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

"""Focused 2-view, 8D EE state, 7D normalized delta-controller unit checks."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from scripts import evaluate_stackcube_pi05 as baseline
from scripts.evaluate_stackcube_pi05_ee_delta import (
    decode_action_chunk, decode_first_action, evaluate_one,
)
from toolkits.lerobot.stackcube_ee_delta import (
    ee_state, extract_ee_observation, normalized_ee_action, policy_input
)

ROOT = Path(__file__).resolve().parents[2]
YAML = ROOT / "examples/sft/config/maniskill_stackcube_panda_sft_openpi_pi05_easy_ee_delta_two_cam.yaml"


def make_scene():
    pose = SimpleNamespace(
        p=np.array([[0.2, -0.1, 0.3]], dtype=np.float32),
        q=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            agent=SimpleNamespace(tcp=SimpleNamespace(pose=pose))
        )
    )
    obs = {
        "agent": {"qpos": np.array([[0, 0, 0, 0, 0, 0, 0, .03, .03]], dtype=np.float32)},
        "sensor_data": {
            "base_camera": {
                "rgb": np.full((1, 224, 224, 3), 40, dtype=np.uint8)
            },
            "hand_camera": {
                "rgb": np.full((1, 224, 224, 3), 210, dtype=np.uint8)
            },
        },
    }
    return env, obs


def test_ee_state_is_world_xyz_quaternion_and_finger_opening():
    env, obs = make_scene()
    result = ee_state(env, obs)
    assert result.shape == (8,)
    np.testing.assert_allclose(
        result, [0.2, -0.1, 0.3, 1.0, 0.0, 0.0, 0.0, 0.06]
    )
    data = extract_ee_observation(env, obs)
    policy = policy_input(data)
    assert policy["main_images"].shape == (1, 224, 224, 3)
    assert policy["wrist_images"].shape == (1, 224, 224, 3)
    assert policy["states"].shape == (1, 8)
    assert policy["task_descriptions"] == [
        "stack the red cube on top of the green cube and release it"
    ]
    assert not np.array_equal(
        policy["main_images"], policy["wrist_images"]
    )


def test_delta_pose_actions_match_normalized_controller_convention():
    from scipy.spatial.transform import Rotation

    q0 = np.array([1, 0, 0, 0], dtype=float)
    q1_xyzw = Rotation.from_euler("XYZ", [0, 0, 0.05]).as_quat()
    q1 = q1_xyzw[[3, 0, 1, 2]]
    action = normalized_ee_action(
        np.zeros(3), q0, np.array([0.02, -0.03, 0.04]),
        q1, -1.0,
    )
    assert action.shape == (7,)
    np.testing.assert_allclose(
        action, [0.2, -0.3, 0.4, 0, 0, -0.5, -1.0], atol=1e-5,
    )


def test_delta_pose_action_clips_large_moves():
    action = normalized_ee_action(
        np.zeros(3), [1, 0, 0, 0], [1, -1, 0], [1, 0, 0, 0], 9.0
    )
    np.testing.assert_allclose(action, [1, -1, 0, 0, 0, 0, 1])


def test_ee_action_evaluator_requires_7d():
    chunk = np.zeros((1, 10, 7), dtype=np.float32)
    chunk[0, 0, 0] = .2
    assert decode_first_action(chunk).shape == (7,)
    assert decode_first_action(chunk)[0] == pytest.approx(.2)
    with pytest.raises(ValueError, match="Expected finite"):
        decode_first_action(np.zeros((1, 10, 8), dtype=np.float32))


def test_new_sft_yaml_and_eval_contract():
    cfg = OmegaConf.load(YAML)
    assert cfg.actor.model.openpi.num_images_in_input == 2
    assert cfg.actor.model.action_dim == 7
    assert cfg.actor.model.openpi.discrete_state_input is True
    assert cfg.actor.model.openpi.config_name == (
        "pi05_stackcube_panda_easy_ee_delta_two_cam_state"
    )
    assert cfg.data.train_data_paths is not None
    model_cfg, _ = baseline.make_model_config(
        YAML, ROOT / "examples/sft/config/model/pi0_5.yaml",
        Path("/tmp/model"), Path("/tmp/norm_stats.json")
    )
    assert model_cfg.action_dim == 7
    assert model_cfg.openpi.num_images_in_input == 2


def test_registered_openpi_data_config_activates_wrist_and_7d():
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    config = get_openpi_config("pi05_stackcube_panda_easy_ee_delta_two_cam_state")
    assert config.model.discrete_state_input is True
    assert config.model.action_horizon == 10
    assert config.data.output_action_dim == 7
    assert config.data.use_wrist_image is True
    assert config.data.extra_delta_transform is False


def test_dataset_adapter_8d_state_7d_actions(monkeypatch):
    from rlinf.data.datasets.openpi.v3_stackcube_dataset import V3StackCubeDataset

    adapter = V3StackCubeDataset.__new__(V3StackCubeDataset)
    adapter.frames = [{
        "actions": [0.0] * 7, "state": [0.0] * 8,
        "episode_index": 0, "frame_index": 0,
    }]
    adapter._last_index = np.array([0], dtype=np.int64)
    adapter.action_horizon = 10
    adapter.use_wrist_image = True
    adapter.state_dim = 8
    adapter.action_dim = 7

    def fake_image(frame, *, image_key="image"):
        return np.full(
            (224, 224, 3),
            10 if image_key == "image" else 210,
            dtype=np.uint8
        )
    monkeypatch.setattr(adapter, "_embedded_image", fake_image)
    sample = adapter[0]
    assert sample["state"].shape == (8,)
    assert sample["actions"].shape == (10, 7)
    assert sample["wrist_image"].shape == (224, 224, 3)
    assert not np.array_equal(sample["image"], sample["wrist_image"])



def test_training_aligned_action_chunk_and_replan_mode():
    chunk = np.zeros((1, 10, 7), dtype=np.float32)
    chunk[0, :, 0] = np.arange(10, dtype=np.float32) / 10
    aligned = decode_action_chunk(chunk)
    assert aligned.shape == (10, 7)
    np.testing.assert_allclose(aligned[:, 0], np.arange(10) / 10)
    assert decode_action_chunk(chunk, action_chunk=1).shape == (1, 7)
    with pytest.raises(ValueError, match="action_chunk"):
        decode_action_chunk(chunk, action_chunk=2)


def test_rollout_executes_predicted_chunk_in_order(monkeypatch):
    """10 predicted EE actions must be executed before resampling the policy."""
    from scripts import evaluate_stackcube_pi05_ee_delta as evaluate

    monkeypatch.setattr(
        evaluate, "extract_ee_observation",
        lambda env, obs: {"image": np.zeros((2, 2, 3), dtype=np.uint8)},
    )
    monkeypatch.setattr(
        evaluate, "policy_input", lambda record: {"dummy": True},
    )

    class StubEnv:
        def __init__(self):
            self.actions = []
            self.unwrapped = SimpleNamespace(
                agent=SimpleNamespace(
                    tcp=SimpleNamespace(
                        pose=SimpleNamespace(p=np.array([[0., 0., .2]]))
                    ),
                    is_grasping=lambda cube: np.array([len(self.actions) >= 3]),
                ),
                cubeA=SimpleNamespace(pose=SimpleNamespace(p=np.array([[0., 0., 0.]]))),
            )

        def reset(self, seed):
            self.actions = []
            return {}, {}

        def step(self, action):
            self.actions.append(np.asarray(action).reshape(7).copy())
            done = len(self.actions) == 10
            return {}, 0.0, np.array([done]), np.array([False]), {
                "success": np.array([False])
            }

    class StubPolicy:
        def __init__(self):
            self.calls = 0

        def predict_action_batch(self, obs, mode):
            self.calls += 1
            predicted = np.zeros((1, 10, 7), dtype=np.float32)
            predicted[0, :, 0] = np.arange(10, dtype=np.float32) / 10
            return predicted, {}

    env = StubEnv()
    policy = StubPolicy()
    result, frames = evaluate_one(
        env, policy, seed=123, max_steps=10,
        record_video=False, action_chunk=10,
    )
    assert policy.calls == 1
    assert len(env.actions) == 10
    np.testing.assert_allclose(
        np.asarray(env.actions)[:, 0], np.arange(10) / 10
    )
    assert result["ever_grasped"] is True
    assert result["action_chunk_executed"] == 10
    assert frames == []


def test_ee_eval_defaults_to_closed_loop_replanning():
    import inspect
    from scripts.evaluate_stackcube_pi05_ee_delta import (
        evaluate_one, decode_action_chunk,
    )

    assert inspect.signature(evaluate_one).parameters["action_chunk"].default == 1
    assert inspect.signature(decode_action_chunk).parameters["action_chunk"].default == 1
