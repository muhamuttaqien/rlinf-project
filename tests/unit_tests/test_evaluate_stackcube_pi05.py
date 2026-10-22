# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Focused CPU tests for ManiSkill StackCube Pi0.5 policy evaluation."""

import numpy as np
import pytest

from scripts import evaluate_stackcube_pi05 as evaluator


def test_action_chunk_keeps_8d_absolute_targets():
    action = np.arange(80, dtype=np.float32).reshape(1, 10, 8)
    result = evaluator.actions_to_numpy(action, chunk=3)
    assert result.shape == (3, 8)
    np.testing.assert_array_equal(result, action[0, :3])


def test_invalid_action_shape_and_nan_are_rejected():
    with pytest.raises(ValueError, match="Expected policy actions"):
        evaluator.actions_to_numpy(np.zeros((10, 7), dtype=np.float32))
    bad = np.zeros((1, 10, 8), dtype=np.float32)
    bad[0, 0, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        evaluator.actions_to_numpy(bad)


def test_success_is_read_from_simulator_not_reward():
    assert evaluator.success_from_info({"success": np.array([True])})
    assert not evaluator.success_from_info({"success": np.array([False])})
    with pytest.raises(KeyError, match="success"):
        evaluator.success_from_info({"reward": 1.0})


def test_rollout_stops_at_success_and_uses_predicted_actions(monkeypatch):
    class FakeModel:
        calls = 0

        def predict_action_batch(self, obs, mode):
            self.calls += 1
            assert mode == "eval"
            assert obs["states"].shape == (1, 9)
            return np.ones((1, 10, 8), dtype=np.float32), {}

    class FakeEnv:
        steps = 0

        def reset(self, *, seed):
            assert seed == 10_000
            return {"frame": 0}, {}

        def step(self, action):
            assert action.shape == (1, 8)
            np.testing.assert_array_equal(action, 1.0)
            self.steps += 1
            return {"frame": self.steps}, 0.0, False, False, {
                "success": np.array([self.steps == 3])
            }

    def fake_extract(obs):
        frame = np.full((8, 8, 3), obs["frame"], dtype=np.uint8)
        return {"states": np.zeros((1, 9)), "main_images": frame[None]}, frame

    monkeypatch.setattr(evaluator, "extract_stackcube_obs", fake_extract)
    env = FakeEnv()
    model = FakeModel()
    metrics, frames = evaluator.evaluate_episode(
        env, model, seed=10_000, max_steps=300,
        record_video=True, video_stride=2,
    )
    assert metrics == {"seed": 10_000, "success": True, "steps": 3}
    assert model.calls == 1
    assert env.steps == 3
    # Initial state, periodic frame (step 2), then successful final frame
    # (step 3, even though video_stride is 2).
    assert len(frames) == 3
    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 2, 3]


def test_rollout_stops_at_truncation(monkeypatch):
    class FakeModel:
        def predict_action_batch(self, obs, mode):
            return np.zeros((1, 10, 8), dtype=np.float32), {}

    class FakeEnv:
        n = 0

        def reset(self, *, seed):
            return {}, {}

        def step(self, action):
            self.n += 1
            return {}, 1.0, False, self.n == 2, {"success": False}

    monkeypatch.setattr(
        evaluator, "extract_stackcube_obs",
        lambda _obs: ({"states": np.zeros((1, 9))}, np.zeros((8, 8, 3))),
    )
    result, frames = evaluator.evaluate_episode(
        FakeEnv(), FakeModel(), 10_001, max_steps=300, record_video=False,
    )
    assert result == {"seed": 10_001, "success": False, "steps": 2}
    assert frames == []



@pytest.mark.parametrize(
    ("action_chunk", "expected_calls", "expected_arm_targets"),
    [
        (10, 1, [1.0, 2.0, 3.0]),
        (1, 3, [1.0, 101.0, 201.0]),
    ],
)
def test_replanning_interval_executes_only_requested_actions(
    monkeypatch, action_chunk, expected_calls, expected_arm_targets
):
    """One-action mode must reobserve at every step, not replay a cached chunk."""
    class FakeModel:
        calls = 0

        def predict_action_batch(self, obs, mode):
            assert mode == "eval"
            self.calls += 1
            observed_step = int(obs["states"][0, 0])
            values = np.arange(1, 11, dtype=np.float32) + observed_step * 100
            return np.broadcast_to(values[None, :, None], (1, 10, 8)).copy(), {}

    class FakeEnv:
        steps = 0

        def __init__(self):
            self.commands = []

        def reset(self, *, seed):
            assert seed == 10_000
            return {"frame": 0}, {}

        def step(self, action):
            assert action.shape == (1, 8)
            self.commands.append(float(action[0, 0]))
            self.steps += 1
            return (
                {"frame": self.steps},
                0.0,
                False,
                False,
                {"success": np.array([False])},
            )

    def fake_extract(obs):
        step = obs["frame"]
        return {
            "states": np.full((1, 9), step, dtype=np.float32),
            "main_images": np.zeros((1, 8, 8, 3), dtype=np.uint8),
        }, np.zeros((8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(evaluator, "extract_stackcube_obs", fake_extract)
    env, model = FakeEnv(), FakeModel()
    metrics, frames = evaluator.evaluate_episode(
        env, model, seed=10_000, max_steps=3,
        record_video=False, action_chunk=action_chunk,
    )
    assert metrics == {"seed": 10_000, "success": False, "steps": 3}
    assert model.calls == expected_calls
    assert env.commands == expected_arm_targets
    assert frames == []


@pytest.mark.parametrize("chunk", [0, 11])
def test_invalid_replanning_interval_rejected_before_reset(chunk):
    with pytest.raises(ValueError, match="action_chunk"):
        evaluator.evaluate_episode(
            None, None, seed=0, max_steps=1,
            record_video=False, action_chunk=chunk,
        )
