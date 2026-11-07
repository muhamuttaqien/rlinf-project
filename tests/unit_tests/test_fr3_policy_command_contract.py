"""Offline-only tests of candidate FR3 policy command review (no ROS import)."""

from dataclasses import replace

import pytest

from toolkits.realworld_check.fr3_policy_command_contract import (
    CandidateAction, PreviewOwner, ReviewLimits, dispatch_to_ros,
    review_candidate_action,
)


def fixture():
    action = CandidateAction(
        values=(0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.1),
        owner="offline_unit_test", sequence=2,
        source_observation_monotonic_ns=900_000_000,
        issued_monotonic_ns=950_000_000, action_interval_ns=20_000_000,
    )
    limits = ReviewLimits(
        max_translation_delta_m=0.005,
        max_rotation_delta_rad=0.02,
        gripper_min_candidate_rad=0.0,
        gripper_max_candidate_rad=0.5,
        max_observation_age_ns=200_000_000,
        max_request_age_ns=100_000_000,
        max_action_interval_ns=30_000_000,
    )
    options = dict(
        limits=limits,
        preview_owner=PreviewOwner("offline_unit_test", 1_100_000_000),
        now_monotonic_ns=1_000_000_000,
        last_sequence=1,
    )
    return action, options


def assert_blocked(report):
    assert report["real_robot_execution_approved"] is False
    assert report["policy_output_mapped"] is False
    assert report["real_controller_authority_granted"] is False
    assert report["commands_created"] == 0
    assert report["commands_sent"] == 0


def test_synthetic_fixture_passes_review_but_never_authorizes():
    action, kwargs = fixture()
    report = review_candidate_action(action, **kwargs)
    assert report["review_checks_passed"] is True
    assert report["review_rejections"] == []
    assert_blocked(report)


def test_missing_bounds_fail_closed():
    action, kwargs = fixture()
    report = review_candidate_action(action, **{**kwargs, "limits": None})
    assert not report["review_checks_passed"]
    assert "Explicit review-only bounds required; no default limits" in report["review_rejections"]
    assert_blocked(report)


def test_stackcube_origin_cannot_be_reinterpreted():
    action, kwargs = fixture()
    report = review_candidate_action(replace(action, origin="pi05_stackcube_predicted"), **kwargs)
    assert not report["review_checks_passed"]
    assert_blocked(report)


@pytest.mark.parametrize("field", ["frame", "rotation_frame", "clock_domain", "schema"])
def test_mismatched_contract_rejected(field):
    action, kwargs = fixture()
    report = review_candidate_action(replace(action, **{field: "unverified_other"}), **kwargs)
    assert not report["review_checks_passed"]
    assert_blocked(report)


@pytest.mark.parametrize("values", [
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.1),
    (0.006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1),
    (0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.1),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6),
])
def test_invalid_values_or_bounds_rejected(values):
    action, kwargs = fixture()
    report = review_candidate_action(replace(action, values=values), **kwargs)
    assert not report["review_checks_passed"]
    assert_blocked(report)


def test_replay_rejected():
    action, kwargs = fixture()
    report = review_candidate_action(action, **{**kwargs, "last_sequence": 2})
    assert not report["review_checks_passed"]
    assert_blocked(report)


@pytest.mark.parametrize("changed", [
    {"issued_monotonic_ns": 1_000_000_001},
    {"source_observation_monotonic_ns": 500_000_000},
    {"issued_monotonic_ns": 800_000_000},
    {"action_interval_ns": 50_000_000},
])
def test_stale_future_or_interval_rejected(changed):
    action, kwargs = fixture()
    report = review_candidate_action(replace(action, **changed), **kwargs)
    assert not report["review_checks_passed"]
    assert_blocked(report)


def test_wrong_owner_or_expired_preview_rejected():
    action, kwargs = fixture()
    for owner in (
        PreviewOwner("different", 1_100_000_000),
        PreviewOwner("offline_unit_test", 1_000_000_000),
    ):
        report = review_candidate_action(action, **{**kwargs, "preview_owner": owner})
        assert not report["review_checks_passed"]
        assert_blocked(report)


def test_no_dispatch_even_after_valid_preview():
    action, kwargs = fixture()
    assert review_candidate_action(action, **kwargs)["review_checks_passed"]
    with pytest.raises(PermissionError, match="dispatch unavailable"):
        dispatch_to_ros(action, enabled=True)


def test_no_inferred_ros_interface_import():
    import sys
    import toolkits.realworld_check.fr3_policy_command_contract as contract
    assert not hasattr(contract, "create_twist")
    assert not hasattr(contract, "gripper_goal")
    assert "rclpy" not in contract.__dict__


def test_invalid_limit_types_fail_closed_not_crash():
    action, kwargs = fixture()
    bad = replace(kwargs["limits"], max_observation_age_ns="unknown")
    report = review_candidate_action(action, **{**kwargs, "limits": bad})
    assert not report["review_checks_passed"]
    assert "Invalid explicit review limits" in report["review_rejections"]
    assert_blocked(report)
