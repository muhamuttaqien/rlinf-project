"""CPU-only checks: distinct window starts, bounded matching, no segment joins."""

import numpy as np
import pytest

from toolkits.realworld_check.fr3_request_transition_diagnostic import (
    analyze,
    match_events,
    transition_score,
)


def test_matching_uses_each_event_once():
    count, lags = match_events([4, 5], [4], tolerance=2)
    assert count == 1
    assert lags == [0]


def test_no_cross_segment_transition():
    sequences = [
        (np.array([0, 0, 0], dtype=bool), np.array([0, 0, 0], dtype=bool)),
        (np.array([1, 1, 1], dtype=bool), np.array([1, 1, 1], dtype=bool)),
    ]
    result = transition_score(sequences, direction=True)
    assert result["true_events"] == 0
    assert result["predicted_events"] == 0
    assert result["f1"] is None


def test_analyze_all_windows_uses_first_horizon_only():
    segments = [{"episode": "episode_0001", "start": 0, "stop": 13},
                {"episode": "episode_0002", "start": 20, "stop": 32}]
    # Segment 1 contributes 4 windows, segment 2 contributes 3 windows.
    actual = np.zeros((7, 10, 7), dtype=float)
    pred = actual.copy()
    actual[:, 0, 6] = [0, 0, 1, 1, 0, 1, 1]
    pred[:, 0, 6] = actual[:, 0, 6]
    # These are deliberately wrong horizon steps; they must not affect scoring.
    pred[:, 1:, 6] = 1
    actual[:, 1:, 6] = 0
    report = analyze(pred, actual, segments, tolerance=0)
    assert report["distinct_scored_steps"] == 7
    assert report["tail_frames_omitted"] == 18
    assert report["gripper"]["first_step_accuracy"] == 1
    assert report["gripper"]["open_transition"]["true_events"] == 2
    assert report["gripper"]["open_transition"]["matched_events"] == 2
    assert report["gripper"]["close_transition"]["true_events"] == 0
    assert report["physical_execution_approved"] is False


def test_analyze_rejects_partial_or_misaligned_full_windows():
    with pytest.raises(ValueError, match="All windows required"):
        analyze(np.zeros((2, 10, 7)), np.zeros((2, 10, 7)),
                [{"episode": "episode_0001", "start": 0, "stop": 13}])
