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

"""CPU-only tests for isolated red-cube interventions and short-rollout metrics."""

import numpy as np
import pytest

from scripts import diagnose_stackcube_pi05_controlled_cube as diag


def test_targets_are_distinct_and_cannot_overlap_green_cube():
    green = np.array([0.0, 0.0, 0.02])
    tcp = np.array([0.0, -0.5, 0.3])
    candidates = [
        (0, np.array([0.1, 0.0, 0.02])),
        (1, np.array([0.005, 0.0, 0.02])),  # too close to green
        (2, np.array([0.11, 0.0, 0.02])),   # near-duplicate of seed 0
        (3, np.array([0.0, 0.15, 0.02])),
        (4, np.array([-0.15, 0.0, 0.02])),
        (5, np.array([0.0, -0.15, 0.02])),
        (6, np.array([0.15, 0.15, 0.02])),
    ]
    positions = diag.select_targets(
        candidates, green, tcp, max_scenes=4
    )
    assert [seed for seed, _ in positions] == [0, 3, 4, 5]
    assert len(positions) == 4


def test_summarize_approach_timing_and_initial_action_differences():
    xs = (0.0, 0.10, 0.20, 0.30)
    traces, chunks = [], []
    for i, x in enumerate(xs):
        initial = 0.4
        best = 0.35 - i * 0.08
        traces.append({
            "cubeA_initial_xyz_m": [x, 0, 0.02],
            "initial_tcp_distance_m": initial,
            "final_tcp_distance_m": 0.38,
            "min_tcp_distance_m": best,
            "ever_grasped": i == 3,
            "first_closing_distance_m": 0.16 if i < 3 else 0.07,
        })
        chunk = np.zeros((10, 8), dtype=np.float32)
        chunk[:, :7] = i * 0.01
        chunk[:, 7] = 1.0 - 0.1 * i
        chunks.append(chunk)

    report = diag.summarize(traces, chunks)
    assert report["scene_count"] == 4
    assert report["red_x_span_m"] == pytest.approx(0.3)
    assert report["red_y_span_m"] == pytest.approx(0.0)
    assert report["mean_best_tcp_distance_reduction_m"] == pytest.approx(
        0.4 - np.mean([0.35, 0.27, 0.19, 0.11])
    )
    assert report["scenes_with_tcp_under_8cm"] == 0
    assert report["scenes_with_grasp"] == 1
    assert report["early_closures_over_10cm"] == 3
    assert report["initial_arm_chunk_pairwise_mae_rad"] > 0


def test_summarize_rejects_short_or_mismatched_trials():
    trial = {
        "cubeA_initial_xyz_m": [0, 0, 0.02],
        "initial_tcp_distance_m": 0.3,
        "final_tcp_distance_m": 0.28,
        "min_tcp_distance_m": 0.25,
        "ever_grasped": False,
        "first_closing_distance_m": None,
    }
    chunk = np.zeros((10, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="At least four"):
        diag.summarize([trial] * 3, [chunk] * 3)
    with pytest.raises(ValueError, match="At least four"):
        diag.summarize([trial] * 4, [chunk] * 3)
