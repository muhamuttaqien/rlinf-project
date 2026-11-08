"""Regression tests for LIBERO normalizer width compatibility."""
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.data.datasets.recap.libero_norm_validation import (
    validate_libero_norm_stats,
)


def _make_stats(state_width=32, action_width=32):
    def item(actual_width, padded_width):
        mean = np.zeros(padded_width, dtype=np.float32)
        mean[:actual_width] = np.arange(actual_width, dtype=np.float32)
        return SimpleNamespace(
            mean=mean,
            std=np.concatenate([np.ones(actual_width), np.zeros(padded_width - actual_width)]),
            q01=mean.copy(),
            q99=mean.copy(),
        )

    return {"state": item(8, state_width), "actions": item(7, action_width)}


def test_accepts_official_sft_zero_padded_32_wide_stats():
    assert validate_libero_norm_stats(_make_stats()) == (32, 32)


def test_accepts_native_libero_8_7_stats():
    assert validate_libero_norm_stats(_make_stats(8, 7)) == (8, 7)


def test_rejects_nonzero_padded_stats():
    stats = _make_stats()
    stats["actions"].q99[9] = 0.1
    with pytest.raises(ValueError, match="nonzero values in padded"):
        validate_libero_norm_stats(stats)


def test_rejects_incorrect_stats_width():
    with pytest.raises(ValueError, match="expected 8 or zero-padded 32"):
        validate_libero_norm_stats(_make_stats(9, 7))


def test_rejects_missing_statistics():
    stats = _make_stats()
    stats["state"].q01 = None
    with pytest.raises(ValueError, match="state.q01"):
        validate_libero_norm_stats(stats)
