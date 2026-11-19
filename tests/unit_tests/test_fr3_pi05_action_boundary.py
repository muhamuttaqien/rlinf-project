"""Focused, no-hardware tests for the policy-output quarantine."""

import numpy as np
import pytest

from toolkits.realworld_check.fr3_pi05_action_boundary import (
    quarantine_pi05_prediction,
    to_fr3_controller_command,
)


def test_monitor_only_valid_chunk():
    report = quarantine_pi05_prediction(
        np.zeros((10, 7)), checkpoint="sim/full_weights.pt", source_episode="raw"
    )
    assert report["raw_prediction_shape"] == [10, 7]
    assert report["execution_approved"] is False
    assert report["command_messages_created"] == 0
    assert report["robot_commands_sent"] is False


def test_invalid_channels_rejected():
    with pytest.raises(ValueError, match="horizon, 7"):
        quarantine_pi05_prediction(
            np.ones((5, 6)), checkpoint="sim", source_episode="raw"
        )


def test_nonfinite_values_rejected():
    with pytest.raises(ValueError, match="Nonfinite"):
        quarantine_pi05_prediction(
            np.full((3, 7), np.nan), checkpoint="sim", source_episode="raw"
        )


def test_missing_provenance_rejected():
    with pytest.raises(ValueError, match="Checkpoint provenance"):
        quarantine_pi05_prediction(
            np.zeros((2, 7)), checkpoint="", source_episode="raw"
        )


def test_command_conversion_always_blocked():
    with pytest.raises(PermissionError, match="Blocked"):
        to_fr3_controller_command(np.zeros(7), enable=True)
