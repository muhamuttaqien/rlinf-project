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

"""Focused CPU tests for aligned expert/rollout action diagnostics."""

import numpy as np
import pytest

from scripts import diagnose_stackcube_pi05 as diagnostic


def test_trace_action_preserves_controller_units_and_finger_qpos():
    before = np.array([0.0] * 7 + [0.01, 0.01], dtype=np.float32)
    after = np.array([0.02] * 7 + [0.015, 0.012], dtype=np.float32)
    action = np.array([0.1] * 7 + [-1.0], dtype=np.float32)
    row = diagnostic.trace_action(
        seed=10000, step=3, target=action,
        before=before, after=after, success=True,
    )
    assert row["success"] is True
    assert row["step"] == 3
    assert row["gripper_command"] == -1.0
    assert row["finger_qpos_after"] == pytest.approx([0.015, 0.012])
    assert row["finger_qpos_sum"] == pytest.approx(0.027)
    assert row["arm_target_mae_after"] == pytest.approx(0.08)


def test_invalid_shape_is_rejected():
    with pytest.raises(ValueError, match="Invalid or non-finite"):
        diagnostic.trace_action(
            seed=1, step=1, target=np.zeros(7),
            before=np.zeros(9), after=np.zeros(9), success=False
        )


def test_episode_local_expert_indices():
    result = diagnostic.expert_sample_indices(
        [0, 0, 0, 0, 1, 1, 1], episodes=2, per_episode=3
    )
    assert result == [(0, 0), (0, 1), (0, 3), (1, 4), (1, 5), (1, 6)]


def test_traced_env_forwards_actual_action_unchanged():
    class FakeEnvironment:
        def reset(self, *, seed):
            self.position = np.zeros(9, dtype=np.float32)
            return {"agent": {"qpos": self.position.copy()}}, {}

        def step(self, action):
            assert action.shape == (1, 8)
            assert action[0, -1] == -0.5
            self.position[:7] += 0.02
            self.position[7:] = 0.013
            return (
                {"agent": {"qpos": self.position.copy()}},
                0.0, False, False, {"success": np.array([False])},
            )

    env = diagnostic.TracedEnv(FakeEnvironment())
    env.reset(seed=10000)
    target = np.array([[0.1] * 7 + [-0.5]], dtype=np.float32)
    env.step(target)
    assert len(env.rows) == 1
    assert env.rows[0]["arm_target_mae_after"] == pytest.approx(0.08)
    assert env.rows[0]["finger_qpos_sum"] == pytest.approx(0.026)
    env.reset(seed=10001)
    assert env.rows == []


def test_paired_prediction_on_same_expert_observation(monkeypatch):
    class Dataset:
        def __getitem__(self, idx):
            return {
                "image": np.zeros((224, 224, 3), dtype=np.uint8),
                "state": np.zeros(9, dtype=np.float32),
                "actions": np.tile(
                    np.array([0.2] * 7 + [1.0], dtype=np.float32), (10, 1)
                ),
            }

    class Model:
        def predict_action_batch(self, obs, mode):
            assert obs["main_images"].shape == (1, 224, 224, 3)
            assert obs["states"].shape == (1, 9)
            assert mode == "eval"
            return np.tile(
                np.array([0.1] * 7 + [-1.0], dtype=np.float32),
                (1, 10, 1),
            ), {}

    monkeypatch.setattr(
        diagnostic, "_read_expert_dataset",
        lambda repo_id: (Dataset(), [0, 0, 1, 1]),
    )
    rows = diagnostic.compare_expert_predictions(
        Model(), "local/fake", max_episodes=2, samples_per_episode=1
    )
    assert len(rows) == 2
    assert [r["dataset_index"] for r in rows] == [0, 2]
    assert rows[0]["arm_action_mae"] == pytest.approx(0.1)
    assert rows[0]["gripper_action_abs_error"] == pytest.approx(2.0)


def test_summary_keeps_offline_and_online_statistics_separate():
    expert = [{
        "arm_action_mae": 0.2,
        "gripper_action_abs_error": 1.1,
    }]
    rollout = [{
        "arm_target_mae_after": 0.5, "arm_motion_l2": 0.1,
        "gripper_command": -0.5, "finger_qpos_sum": 0.03,
    }]
    stats = diagnostic.summarize_diagnostics(expert, rollout)
    assert stats["paired_expert_samples"] == 1
    assert stats["paired_gripper_action_abs_error"] == pytest.approx(1.1)
    assert stats["mean_finger_qpos_sum"] == pytest.approx(0.03)


def test_task_status_reads_stackcube_flags_and_geometry():
    class Pose:
        def __init__(self, xyz):
            self.p = np.array([xyz], dtype=np.float32)

    class Actor:
        def __init__(self, xyz):
            self.pose = Pose(xyz)

    class Env:
        def __init__(self):
            self.agent = type("Robot", (), {"tcp": Actor([0.0, 0.0, 0.1])})()
            self.cubeA = Actor([0.0, 0.0, 0.05])
            self.cubeB = Actor([0.2, 0.0, 0.05])

    info = {
        "success": np.array([False]),
        "is_cubeA_grasped": np.array([True]),
        "is_cubeA_on_cubeB": np.array([False]),
        "is_cubeA_static": np.array([True]),
    }
    status = diagnostic.stackcube_task_status(info, Env())
    assert status["is_cubeA_grasped"] is True
    assert status["is_cubeA_on_cubeB"] is False
    assert status["is_cubeA_static"] is True
    assert status["tcp_to_cubeA_m"] == pytest.approx(0.05)
    assert status["cubeA_to_cubeB_xy_m"] == pytest.approx(0.2)


def test_summary_counts_task_stages_without_guessing_missing_metrics():
    expert = []
    rollout = [{
        "step": 1, "arm_target_mae_after": 0.01, "arm_motion_l2": 0.01,
        "gripper_command": 1.0, "finger_qpos_sum": 0.08,
        "is_cubeA_grasped": False, "is_cubeA_on_cubeB": False,
        "is_cubeA_static": True, "tcp_to_cubeA_m": 0.12,
        "cubeA_to_cubeB_xy_m": 0.2,
    }, {
        "step": 2, "arm_target_mae_after": 0.02, "arm_motion_l2": 0.02,
        "gripper_command": -1.0, "finger_qpos_sum": 0.0,
        "is_cubeA_grasped": True, "is_cubeA_on_cubeB": False,
        "is_cubeA_static": False, "tcp_to_cubeA_m": 0.02,
        "cubeA_to_cubeB_xy_m": 0.18,
    }]
    stats = diagnostic.summarize_diagnostics(expert, rollout)
    assert stats["is_cubeA_grasped_steps"] == 1
    assert stats["is_cubeA_grasped_first_step"] == 2
    assert stats["is_cubeA_on_cubeB_steps"] == 0
    assert stats["is_cubeA_on_cubeB_first_step"] is None
    assert stats["is_cubeA_static_steps"] == 1
    assert stats["min_tcp_to_cubeA_m"] == pytest.approx(0.02)
    assert stats["min_cubeA_to_cubeB_xy_m"] == pytest.approx(0.18)

    empty = diagnostic.summarize_diagnostics([], [{
        "step": 1, "arm_target_mae_after": 0.01, "arm_motion_l2": 0.01,
        "gripper_command": -1.0, "finger_qpos_sum": 0.0
    }])
    assert "is_cubeA_grasped_steps" not in empty
