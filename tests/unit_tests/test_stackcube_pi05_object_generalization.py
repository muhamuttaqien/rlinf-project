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

"""CPU-only checks for fixed-state, object-position diagnostics."""

import numpy as np
import pytest

from scripts import diagnose_stackcube_pi05_object_generalization as diag


def fake_scene(seed, cube_x, *, qpos_shift=0.0):
    return {
        "seed": seed,
        "image": np.full((8, 8, 3), seed % 256, dtype=np.uint8),
        "actual_qpos": np.full(9, qpos_shift, dtype=np.float32),
        "tcp_xyz_m": np.array([0.1, 0.0, 0.2]),
        "cubeA_xyz_m": np.array([cube_x, 0.0, 0.02]),
        "cubeB_xyz_m": np.array([0.0, 0.2, 0.02]),
    }


def test_summary_records_real_cube_geometry_and_action_response():
    scenes = [
        fake_scene(0, 0.0),
        fake_scene(1, 0.1),
        fake_scene(2, 0.2, qpos_shift=0.01),
        fake_scene(10000, 0.3),
    ]
    actions = []
    for i in range(4):
        chunk = np.zeros((10, 8), dtype=np.float32)
        chunk[:, :7] = i * 0.02
        chunk[:, 7] = 1.0 - 0.1 * i
        actions.append(chunk)

    report = diag.summarize(scenes, actions, sampling_seed=42)
    assert report["scene_count"] == 4
    assert report["distinct_cubeA_xy_positions"] == 4
    assert report["cubeA_x_span_m"] == pytest.approx(0.3)
    assert report["cubeA_y_span_m"] == 0
    assert len(report["pairs"]) == 6
    assert report["mean_pairwise_cubeA_xy_separation_m"] == pytest.approx(
        (0.1 + 0.2 + 0.3 + 0.1 + 0.2 + 0.1) / 6
    )
    assert report["pairs"][0]["arm_mae_all_10_actions_rad"] == pytest.approx(0.02)
    assert report["pairs"][0]["cubeA_xy_separation_m"] == pytest.approx(0.1)
    assert report["max_actual_qpos_deviation_l2"] == pytest.approx(0.03)
    assert report["scenes"][0]["first_predicted_action_8d"][-1] == 1.0
    assert "toward the cube" in report["interpretation_limit"]


def test_reject_unchanged_cube_positions():
    scenes = [fake_scene(i, 0.1) for i in range(4)]
    actions = [np.zeros((10, 8), dtype=np.float32) for _ in scenes]
    with pytest.raises(ValueError, match="Fewer than three"):
        diag.summarize(scenes, actions, sampling_seed=42)


def test_reject_duplicates_and_bad_action_shapes():
    scenes = [fake_scene(i, i / 10) for i in range(4)]
    actions = [np.zeros((10, 8), dtype=np.float32) for _ in scenes]
    scenes[1]["seed"] = 0
    with pytest.raises(ValueError, match="distinct"):
        diag.summarize(scenes, actions, sampling_seed=42)
    scenes[1]["seed"] = 1
    actions[2] = np.zeros((7, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="Invalid 10x8"):
        diag.summarize(scenes, actions, sampling_seed=42)


def test_pairwise_correlation_is_descriptive_only():
    paired = [
        {"cubeA_xy_separation_m": 0.1, "arm_mae_all_10_actions_rad": 0.01},
        {"cubeA_xy_separation_m": 0.2, "arm_mae_all_10_actions_rad": 0.02},
        {"cubeA_xy_separation_m": 0.3, "arm_mae_all_10_actions_rad": 0.03},
    ]
    assert diag.descriptive_pairwise_corr(paired) == pytest.approx(1.0)
    for pair in paired:
        pair["arm_mae_all_10_actions_rad"] = 0.02
    assert diag.descriptive_pairwise_corr(paired) is None
