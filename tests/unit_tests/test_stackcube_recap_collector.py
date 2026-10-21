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

"""Lightweight contract tests for the StackCube RECAP expert collector."""

import importlib.util
from pathlib import Path

import numpy as np


def _collector():
    script = Path(__file__).parents[2] / "toolkits/lerobot/collect_maniskill_stackcube_recap.py"
    spec = importlib.util.spec_from_file_location("stackcube_recap_collector", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stackcube_record_contract():
    collector = _collector()
    image = np.ones((4, 4, 3), dtype=np.uint8)
    obs = {
        "agent": {"qpos": np.arange(9, dtype=np.float32)[None]},
        "sensor_data": {"base_camera": {"rgb": image[None]}},
    }
    frame = collector._extract_observation(obs)
    assert frame["state"].shape == (9,)
    assert frame["image"].shape == (4, 4, 3)
    assert np.array_equal(frame["wrist_image"], frame["image"])
    assert collector._solver_succeeded(({"success": np.array([True])},))
    assert not collector._solver_succeeded(-1)


def test_stackcube_solver_hooks_unwrapped_steps():
    collector = _collector()
    image = np.ones((4, 4, 3), dtype=np.uint8)
    obs = {
        "agent": {"qpos": np.arange(9, dtype=np.float32)[None]},
        "sensor_data": {"base_camera": {"rgb": image[None]}},
    }

    class FakeEnv:
        unwrapped = None

        def reset(self, seed):
            return obs, {}

        def step(self, action):
            return obs, 1.0, False, False, {"success": True}

    env = FakeEnv()
    env.unwrapped = env
    original_reset = env.reset.__func__
    original_step = env.step.__func__

    def fake_solver(e, seed, debug, vis):
        e.reset(seed=seed)
        e.unwrapped.step(np.zeros((1, 8), dtype=np.float32))
        return ({"success": np.array([True])},)

    frames = collector._collect_episode(env, fake_solver, 0)
    assert len(frames) == 1
    assert frames[0]["actions"].shape == (8,)
    assert frames[0]["is_success"] is True
    assert env.reset.__func__ is original_reset
    assert env.step.__func__ is original_step
