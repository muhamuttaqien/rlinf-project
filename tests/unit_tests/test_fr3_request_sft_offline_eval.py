"""CPU-only tests for offline FR3 request checkpoint-evaluation metrics."""

import numpy as np
import pytest

from toolkits.realworld_check.fr3_request_sft_offline_eval import (
    metric_summary, select_windows,
)


def test_select_windows_balances_source_episodes_and_respects_boundaries():
    segments = [
        {"episode": "episode_000", "start": 30, "stop": 50},
        {"episode": "episode_001", "start": 8, "stop": 23},
        {"episode": "episode_001", "start": 35, "stop": 55},
    ]
    selected, total = select_windows(segments, horizon=10, per_episode=2)
    assert total == 11 + 6 + 11
    assert selected == [
        (0, "episode_000"), (10, "episode_000"),
        (11, "episode_001"), (27, "episode_001"),
    ]
    all_indices, _ = select_windows(segments, horizon=10, all_windows=True)
    assert len(all_indices) == total
    assert len({i for i, _ in all_indices}) == total


def test_metric_summary_perfect_requests():
    target = np.zeros((2, 10, 7), dtype=float)
    target[0, :, :6] = 0.25
    target[1, :, :6] = -0.4
    target[0, :, 6] = 1.0
    report = metric_summary(target.copy(), target.copy())
    assert report["arm_6d_rmse"] == pytest.approx(0)
    assert report["gripper_open"]["accuracy"] == 1.0
    assert report["gripper_open"]["f1"] == 1.0


def test_metric_summary_rejects_invalid_true_request():
    bad = np.zeros((1, 10, 7), dtype=float)
    bad[:, :, 6] = 0.3
    with pytest.raises(ValueError, match="binary gripper"):
        metric_summary(bad, bad)
