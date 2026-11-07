# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Offline FR3/Robotiq candidate-action contract; deliberately non-actuating.

An upstream StackCube pi0.5 prediction is *not* an FR3 action. This module
only validates explicitly typed synthetic review fixtures; it does NOT decode
model output, convert 7D vectors to ROS Twist, create gripper goals, acquire
real controller authority, or connect to ROS. It always blocks dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA = "fr3_robotiq_candidate_tcp_delta_v1"
ORIGIN = "synthetic_contract_fixture_only"
FRAME = "fr3_base_candidate_unverified"
ROTATION_FRAME = "base_frame_rotation_vector_candidate_unverified"
CLOCK = "single_host_monotonic_ns"
CHANNEL_NAMES = (
    "delta_tcp_x_m", "delta_tcp_y_m", "delta_tcp_z_m",
    "delta_tcp_rotvec_x_rad", "delta_tcp_rotvec_y_rad",
    "delta_tcp_rotvec_z_rad", "gripper_target_knuckle_rad",
)
UNVERIFIED_GATES = (
    "No trained and FR3-calibrated 19D/7D policy available",
    "Model action channels not proven to match the candidate physical meaning",
    "Controller accepts normalized 6D SpaceMouse Twist, not 7D metric deltas",
    "Franka TCP frame, tool offset, gripper command semantics and safety limits unapproved",
    "No dedicated exclusive ROS publisher, controller interlock or cross-domain authority",
    "No proven cross-host timebase mapping or policy command-loss behavior",
    "Physical safety review and operator approval outstanding",
)


@dataclass(frozen=True)
class CandidateAction:
    """Review-only data, NOT an instruction to the robot.

    Delta translation is represented in metres and rotation vector in radians,
    both *proposed* base-frame semantics. Index 6 is an unverified Robotiq
    knuckle target in radians, NOT the existing action server's accepted units.
    """
    values: Any
    owner: str
    sequence: int
    source_observation_monotonic_ns: int
    issued_monotonic_ns: int
    action_interval_ns: int
    origin: str = ORIGIN
    schema: str = SCHEMA
    frame: str = FRAME
    rotation_frame: str = ROTATION_FRAME
    clock_domain: str = CLOCK


@dataclass(frozen=True)
class ReviewLimits:
    """Caller-supplied *test fixtures*, not certified controller limits."""
    max_translation_delta_m: float
    max_rotation_delta_rad: float
    gripper_min_candidate_rad: float
    gripper_max_candidate_rad: float
    max_observation_age_ns: int
    max_request_age_ns: int
    max_action_interval_ns: int


@dataclass(frozen=True)
class PreviewOwner:
    """Only an offline logical comparison, NOT real controller ownership."""
    name: str
    expires_monotonic_ns: int


def review_candidate_action(
    action: CandidateAction,
    *,
    limits: ReviewLimits | None = None,
    preview_owner: PreviewOwner | None = None,
    now_monotonic_ns: int | None = None,
    last_sequence: int | None = None,
) -> dict[str, Any]:
    """Validate a synthetic contract fixture; never return a ROS command.

    No implicit limits, automatic clipping, fallback frames, inferred physical
    gripper widths, clock offset conversions, or 'enable' argument exist.
    A passing review reports ONLY synthetic interface consistency.
    """
    if not isinstance(action, CandidateAction):
        raise TypeError("Expected CandidateAction")
    reasons: list[str] = []

    if action.origin != ORIGIN:
        reasons.append("Only synthetic fixtures may use this contract; model predictions remain quarantined")
    if action.schema != SCHEMA:
        reasons.append("Action schema mismatch")
    if action.frame != FRAME or action.rotation_frame != ROTATION_FRAME:
        reasons.append("Frame/rotation-frame convention mismatch")
    if action.clock_domain != CLOCK:
        reasons.append("Clock domain is not the declared same-host monotonic timebase")
    if not action.owner or not isinstance(action.owner, str):
        reasons.append("Missing preview owner")
    if type(action.sequence) is not int or action.sequence <= 0:
        reasons.append("Invalid positive sequence")
    if last_sequence is None or type(last_sequence) is not int or last_sequence < 0:
        reasons.append("Previous sequence required for offline replay check")
    elif type(action.sequence) is int and action.sequence <= last_sequence:
        reasons.append("Duplicate or replayed sequence")

    try:
        values = np.asarray(action.values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        values = np.array([])
    if values.shape != (7,) or not np.isfinite(values).all():
        reasons.append("Expected seven finite candidate channel values")

    limits_valid = False
    if limits is None:
        reasons.append("Explicit review-only bounds required; no default limits")
    else:
        try:
            positive = (
                np.isfinite(limits.max_translation_delta_m)
                and 0 < limits.max_translation_delta_m
                and np.isfinite(limits.max_rotation_delta_rad)
                and 0 < limits.max_rotation_delta_rad
                and np.isfinite(limits.gripper_min_candidate_rad)
                and np.isfinite(limits.gripper_max_candidate_rad)
                and limits.gripper_min_candidate_rad < limits.gripper_max_candidate_rad
                and all(
                    type(x) is int and x > 0
                    for x in (
                        limits.max_observation_age_ns,
                        limits.max_request_age_ns,
                        limits.max_action_interval_ns,
                    )
                )
            )
        except (TypeError, ValueError):
            positive = False
        limits_valid = bool(positive)
        if not limits_valid:
            reasons.append("Invalid explicit review limits")
        elif values.shape == (7,) and np.isfinite(values).all():
            if float(np.linalg.norm(values[:3])) > limits.max_translation_delta_m:
                reasons.append("Candidate translation norm exceeds review bound")
            if float(np.linalg.norm(values[3:6])) > limits.max_rotation_delta_rad:
                reasons.append("Candidate rotation norm exceeds review bound")
            if not limits.gripper_min_candidate_rad <= values[6] <= limits.gripper_max_candidate_rad:
                reasons.append("Candidate gripper scalar exceeds review bound")

    if now_monotonic_ns is None or type(now_monotonic_ns) is not int or now_monotonic_ns <= 0:
        reasons.append("Explicit positive same-host monotonic time required")
    elif not (
        type(action.source_observation_monotonic_ns) is int
        and type(action.issued_monotonic_ns) is int
        and type(action.action_interval_ns) is int
        and 0 < action.source_observation_monotonic_ns
        <= action.issued_monotonic_ns
        <= now_monotonic_ns
        and action.action_interval_ns > 0
    ):
        reasons.append("Observation/issue time invalid, future-dated or non-monotonic")
    elif limits_valid:
        if now_monotonic_ns - action.source_observation_monotonic_ns > limits.max_observation_age_ns:
            reasons.append("Observation too old")
        if now_monotonic_ns - action.issued_monotonic_ns > limits.max_request_age_ns:
            reasons.append("Request too old")
        if action.action_interval_ns > limits.max_action_interval_ns:
            reasons.append("Action interval exceeds review bound")

    if preview_owner is None or not isinstance(preview_owner, PreviewOwner):
        reasons.append("Explicit offline preview owner required")
    elif (
        not isinstance(preview_owner.name, str)
        or not preview_owner.name
        or action.owner != preview_owner.name
        or type(preview_owner.expires_monotonic_ns) is not int
        or now_monotonic_ns is None
        or type(now_monotonic_ns) is not int
        or now_monotonic_ns >= preview_owner.expires_monotonic_ns
    ):
        reasons.append("Preview owner mismatch or expired (NOT real authority)")

    return {
        "contract": SCHEMA,
        "mode": "offline_review_only",
        "channel_names": list(CHANNEL_NAMES),
        "review_checks_passed": not reasons,
        "review_rejections": reasons,
        "physical_frame_verified": False,
        "hardware_limit_certified": False,
        "real_controller_authority_granted": False,
        "arm_controller_state_last_observed": "inactive",
        "policy_output_mapped": False,
        "real_robot_execution_approved": False,
        "commands_created": 0,
        "commands_sent": 0,
        "remaining_physical_gates": list(UNVERIFIED_GATES),
    }


def dispatch_to_ros(*_args: Any, **_kwargs: Any) -> None:
    """No bypass: dispatch is intentionally impossible in this implementation."""
    raise PermissionError(
        "FR3 policy dispatch unavailable: candidate review is not physical "
        "authorization. Use no publisher or Robotiq goal from this module."
    )
