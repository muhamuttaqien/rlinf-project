"""CPU-only offline temporal-control diagnostic tests."""

import numpy as np
import pytest

from toolkits.realworld_check.fr3_request_temporal_consistency import summarize


def _segments():
    return [
        {"episode": "episode_0001", "start": 0, "stop": 13},
        {"episode": "episode_0002", "start": 20, "stop": 32},
    ]


def test_all_identical_stable_sequences_have_no_sample_sensitivity():
    data = np.zeros((7, 10, 7), dtype=float)
    data[:, :, 6] = 1
    result = summarize(data, data, data, data, _segments())
    assert result["distinct_window_starts"] == 7
    assert result["source_label_self_consistency"]["all_overlaps_consistent"]
    assert result["sampling_sensitivity"]["two_independent_seeds_same_observations"]["linear_x"]["first_step_inter_sample_rmse"] == 0
    assert result["first_step_activity_dynamics"]["target"]["gripper_open"]["activity_state_transitions"] == 0
    assert result["physical_execution_approved"] is False


def test_overlap_detects_inconsistent_future_targets():
    data = np.zeros((7, 10, 7), dtype=float)
    data[1, 0, 0] = 0.6
    result = summarize(data, data, data, data, _segments())
    assert not result["source_label_self_consistency"]["all_overlaps_consistent"]
    assert result["source_label_self_consistency"]["mismatched_vectors_at_tolerance"] >= 1


def test_sampling_difference_detected_without_model_or_ros():
    target = np.zeros((7, 10, 7), dtype=float)
    a = target.copy()
    b = target.copy()
    b[:, 0, 4] = 0.3
    result = summarize(a, b, a, target, _segments())
    assert result["sampling_sensitivity"]["two_independent_seeds_same_observations"]["angular_y"]["first_step_inter_sample_rmse"] == pytest.approx(0.3)
    assert result["first_step_activity_dynamics"]["independent_seed_b"]["angular_y"]["activity_state_transitions"] == 0
    assert result["prediction_performance"]["independent_seed_b"]["axes"]["angular_y"]["rmse"] == pytest.approx(0.3)


def test_requires_complete_strict_windows():
    data = np.zeros((6, 10, 7), dtype=float)
    with pytest.raises(ValueError, match="all strict windows"):
        summarize(data, data, data, data, _segments())
