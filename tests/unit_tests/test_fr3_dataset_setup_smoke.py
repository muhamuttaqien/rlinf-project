"""CPU-only FR3 clock-alignment checks; no robot, ROS, or LeRobot dependency."""

import pytest

from toolkits.realworld_check.fr3_dataset_setup_smoke import (
    align_by_calibrated_clock,
    synthetic_alignment_smoke,
)


def test_synthetic_alignment_passes():
    assert synthetic_alignment_smoke()["passed"] is True


def test_missing_offset_rejected():
    with pytest.raises(ValueError, match="measured integer clock"):
        align_by_calibrated_clock([100], [101],
                                  nuc_minus_workstation_ns=None,
                                  uncertainty_ns=0)


def test_unordered_stamps_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        align_by_calibrated_clock([100, 90], [101, 110],
                                  nuc_minus_workstation_ns=0,
                                  uncertainty_ns=0)


def test_wrong_clock_offset_rejected():
    with pytest.raises(ValueError, match="skew tolerance"):
        align_by_calibrated_clock([1_000_000_000],
                                  [3_000_000_000],
                                  nuc_minus_workstation_ns=0,
                                  uncertainty_ns=1_000_000)


def test_uncertainty_must_fit_budget():
    with pytest.raises(ValueError, match="uncertainty"):
        align_by_calibrated_clock([1_000_000_000],
                                  [1_000_000_000],
                                  nuc_minus_workstation_ns=0,
                                  uncertainty_ns=20_000_000)


def test_no_duplicate_nearest_ticks():
    with pytest.raises(ValueError, match="one/out-of-order"):
        align_by_calibrated_clock([1_000_000_000, 1_000_000_001],
                                  [1_000_000_000, 1_100_000_000],
                                  nuc_minus_workstation_ns=0,
                                  uncertainty_ns=0)
