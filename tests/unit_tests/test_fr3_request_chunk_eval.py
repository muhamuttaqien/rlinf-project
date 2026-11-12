"""CPU-only tests for the non-overlapping FR3 request chunk evaluation."""

import numpy as np

from toolkits.realworld_check.fr3_request_sft_offline_eval import (
    chunk_transition_summary,
    select_chunk_windows,
)


def test_select_chunks_uses_distinct_steps_and_segment_boundaries():
    segments = [
        {"episode": "episode_0001", "start": 10, "stop": 33},
        {"episode": "episode_0002", "start": 40, "stop": 60},
    ]
    selected, total = select_chunk_windows(segments)
    assert total == 25
    assert selected == [
        (0, "episode_0001"), (10, "episode_0001"),
        (14, "episode_0002"), (24, "episode_0002"),
    ]


def test_chunk_events_do_not_cross_episode_boundaries():
    segments = [
        {"episode": "episode_0001", "start": 0, "stop": 19},
        {"episode": "episode_0002", "start": 20, "stop": 39},
    ]
    actual = np.zeros((2, 10, 7))
    predicted = actual.copy()
    actual[1, :, 6] = 1.0
    predicted[1, :, 6] = 1.0
    summary = chunk_transition_summary(predicted, actual, segments)
    assert summary["evaluated_steps"] == 20
    assert summary["segments_kept_separate"] == 2
    assert summary["gripper_open"]["true_events"] == 0
    assert summary["gripper_open"]["predicted_events"] == 0
