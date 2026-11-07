"""Offline FR3 handoff preview: strictly no ROS or real actuator calls."""

from dataclasses import replace

import pytest

from toolkits.realworld_check.fr3_policy_command_contract import (
    CandidateAction, ReviewLimits,
)
from toolkits.realworld_check.fr3_policy_handoff_preview import (
    ControllerReviewScale, OfflinePolicyHandoff,
)


def setup_preview():
    limits = ReviewLimits(
        max_translation_delta_m=0.005, max_rotation_delta_rad=0.02,
        gripper_min_candidate_rad=0.0, gripper_max_candidate_rad=0.8,
        max_observation_age_ns=200_000_000,
        max_request_age_ns=100_000_000,
        max_action_interval_ns=30_000_000,
    )
    # Observed *source* scaling, NOT physically certified.
    scale = ControllerReviewScale(0.007, 0.03, 50_000_000)
    preview = OfflinePolicyHandoff(limits, scale)
    preview.arm_preview("synthetic", now_ns=900_000_000, expires_ns=1_100_000_000)
    action = CandidateAction(
        values=(0.0035, 0, 0, 0, 0.015, 0, 0.0),
        owner="synthetic", sequence=1,
        source_observation_monotonic_ns=900_000_000,
        issued_monotonic_ns=950_000_000,
        action_interval_ns=20_000_000,
    )
    return preview, action


def test_normalized_math_preview_has_no_ros_command():
    handoff, action = setup_preview()
    handoff.heartbeat("synthetic", now_ns=945_000_000)
    result = handoff.preview(action, now_ns=980_000_000)
    assert result["illustrative_normalized_axes"] == [0.5, 0, 0, 0, 0.5, 0]
    assert result["illustrative_gripper_event"] == "open_preview"
    assert result["ros_messages_created"] == 0
    assert result["robot_commands_sent"] == 0
    assert result["physical_execution_approved"] is False


def test_binary_close_event_is_only_a_preview():
    handoff, action = setup_preview()
    handoff.heartbeat("synthetic", now_ns=945_000_000)
    result = handoff.preview(replace(action, values=(0, 0, 0, 0, 0, 0, 0.7929)),
                             now_ns=980_000_000)
    assert result["illustrative_gripper_event"] == "close_preview"


def test_continuous_gripper_target_latches_off():
    handoff, action = setup_preview()
    handoff.heartbeat("synthetic", now_ns=945_000_000)
    with pytest.raises(PermissionError, match="Intermediate Robotiq"):
        handoff.preview(replace(action, values=(0, 0, 0, 0, 0, 0, 0.2)),
                        now_ns=980_000_000)
    assert handoff.state == "fault_latched"
    with pytest.raises(PermissionError, match="fault latched"):
        handoff.preview(action, now_ns=990_000_000)


def test_out_of_range_no_clipping():
    handoff, action = setup_preview()
    handoff.heartbeat("synthetic", now_ns=945_000_000)
    with pytest.raises(PermissionError, match="normalized controller-input"):
        handoff.preview(replace(action, values=(0, 0, 0, 0, 0, 0.035, 0)),
                        now_ns=980_000_000)
    assert handoff.state == "fault_latched"


def test_owner_mismatch_latches_off():
    handoff, action = setup_preview()
    with pytest.raises(PermissionError, match="Owner"):
        handoff.heartbeat("other", now_ns=945_000_000)
    assert handoff.state == "fault_latched"


def test_expired_heartbeat_latches_off():
    handoff, action = setup_preview()
    with pytest.raises(PermissionError, match="heartbeat"):
        handoff.preview(action, now_ns=980_000_000)
    assert handoff.state == "fault_latched"


def test_replay_and_wrong_origin_latch_off():
    handoff, action = setup_preview()
    handoff.heartbeat("synthetic", now_ns=945_000_000)
    handoff.preview(action, now_ns=980_000_000)
    with pytest.raises(PermissionError, match="Candidate rejected"):
        handoff.preview(action, now_ns=985_000_000)
    assert handoff.state == "fault_latched"
    handoff.disable()
    handoff.arm_preview("synthetic", now_ns=990_000_000, expires_ns=1_100_000_000)
    handoff.heartbeat("synthetic", now_ns=991_000_000)
    with pytest.raises(PermissionError, match="Candidate rejected"):
        handoff.preview(replace(action, origin="stackcube_policy"),
                        now_ns=992_000_000)


def test_preview_disabled_prevents_command_and_arm():
    handoff, action = setup_preview()
    with pytest.raises(PermissionError, match="not implemented"):
        handoff.send_to_robot(action)
    with pytest.raises(PermissionError, match="disabled before"):
        handoff.arm_preview("second", now_ns=910_000_000,
                            expires_ns=1_100_000_000)
    handoff.disable()
    with pytest.raises(PermissionError, match="disabled"):
        handoff.preview(action, now_ns=980_000_000)


@pytest.mark.parametrize("bad_scale", [
    (0.0, 0.03, 50_000_000),
    (0.007, float("nan"), 50_000_000),
    (0.007, 0.03, 0),
])
def test_invalid_profile_never_arms(bad_scale):
    limits = setup_preview()[0].limits
    with pytest.raises(ValueError):
        OfflinePolicyHandoff(limits, ControllerReviewScale(*bad_scale))
