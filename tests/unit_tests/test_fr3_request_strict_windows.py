"""Pure tests of strict complete-horizon SpaceMouse SFT indexing."""

import pytest

from rlinf.data.datasets.openpi.official_sft_data_loader import valid_horizon_starts


def test_episode_boundary_padding_excluded():
    assert valid_horizon_starts([0, 12], [12, 23], 10, 23) == [0, 1, 2, 12, 13]


def test_short_episodes_excluded():
    assert valid_horizon_starts([0, 5], [5, 17], 10, 17) == [5, 6, 7]


def test_empty_or_overlapping_metadata_rejected():
    with pytest.raises(ValueError):
        valid_horizon_starts([0, 5], [10, 16], 10, 16)
    with pytest.raises(ValueError):
        valid_horizon_starts([0], [5], 10, 5)


def test_horizon_one_includes_all():
    assert valid_horizon_starts([0], [3], 1, 3) == [0, 1, 2]
