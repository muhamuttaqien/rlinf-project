# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""OFFLINE ONLY: candidate FR3 policy-to-controller preview with no ROS transport.

This is a mathematical illustration and fail-closed state machine, NOT a
physically verified controller mapping or an executable bridge. It imports no
ROS modules, never publishes a Twist, sends a gripper goal, or enables control.
The deployed controller uses normalized SpaceMouse intent, and its true
frame/rate/authority semantics require separate NUC-side qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from toolkits.realworld_check.fr3_policy_command_contract import (
    CandidateAction, PreviewOwner, ReviewLimits, review_candidate_action,
)

GRIPPER_OPEN_CANDIDATE_RAD = 0.0
GRIPPER_CLOSED_CANDIDATE_RAD = 0.7929


@dataclass(frozen=True)
class ControllerReviewScale:
    """Explicit, SOURCE-REVIEWED step sizes; NOT certified safety limits.

    An input "Twist" channel denotes a pose-update intent rather than
    metric velocity. Mapping a rotation-vector to controller rotation axes is
    only an *illustrative approximation*, not a proven equivalence.
    """
    translation_per_unit_m: float
    rotation_per_unit_rad: float
    max_heartbeat_age_ns: int

    def validate(self) -> None:
        for value in (self.translation_per_unit_m, self.rotation_per_unit_rad):
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError("Review scale must contain finite positive steps")
        if type(self.max_heartbeat_age_ns) is not int or self.max_heartbeat_age_ns <= 0:
            raise ValueError("Explicit positive heartbeat age required")


class OfflinePolicyHandoff:
    """Single-owner preview, explicit disabled/armed/fault states, NO dispatch.

    No backend address, publisher, action client, or runtime-enable flag
    exists. Validation must not be interpreted as controller ownership.
    """

    def __init__(self, limits: ReviewLimits, scale: ControllerReviewScale):
        scale.validate()
        self.limits = limits
        self.scale = scale
        self._state = "disabled"
        self._owner: PreviewOwner | None = None
        self._heartbeat_ns: int | None = None
        self._last_now_ns = 0
        self._last_sequence = 0

    @property
    def state(self) -> str:
        return self._state

    def _clock(self, now_ns: int) -> None:
        if type(now_ns) is not int or now_ns <= self._last_now_ns:
            self._state = "fault_latched"
            raise ValueError("Preview clock must be positive and strictly increasing")
        self._last_now_ns = now_ns

    def arm_preview(self, owner: str, *, now_ns: int, expires_ns: int) -> None:
        if self._state != "disabled":
            raise PermissionError("Preview must be disabled before an explicit arm")
        self._clock(now_ns)
        if (not isinstance(owner, str) or not owner.strip()
                or type(expires_ns) is not int or expires_ns <= now_ns):
            self._state = "fault_latched"
            raise ValueError("Explicit offline owner and future lease expiry required")
        self._owner = PreviewOwner(owner, expires_ns)
        self._heartbeat_ns = now_ns
        self._last_sequence = 0
        self._state = "armed_preview_only"

    def _check_live(self, owner: str, now_ns: int) -> None:
        if self._state != "armed_preview_only":
            raise PermissionError("Preview disabled or fault latched")
        self._clock(now_ns)
        if (self._owner is None or owner != self._owner.name
                or now_ns >= self._owner.expires_monotonic_ns
                or self._heartbeat_ns is None
                or now_ns - self._heartbeat_ns > self.scale.max_heartbeat_age_ns):
            self._state = "fault_latched"
            raise PermissionError("Owner, lease or heartbeat invalid; preview latched off")

    def heartbeat(self, owner: str, *, now_ns: int) -> None:
        self._check_live(owner, now_ns)
        self._heartbeat_ns = now_ns

    def preview(self, action: CandidateAction, *, now_ns: int) -> dict[str, Any]:
        if not isinstance(action, CandidateAction):
            self._state = "fault_latched"
            raise TypeError("Explicit offline CandidateAction required")
        self._check_live(action.owner, now_ns)
        checked = review_candidate_action(
            action,
            limits=self.limits,
            preview_owner=self._owner,
            now_monotonic_ns=now_ns,
            last_sequence=self._last_sequence,
        )
        if not checked["review_checks_passed"]:
            self._state = "fault_latched"
            raise PermissionError(
                "Candidate rejected: " + "; ".join(checked["review_rejections"])
            )

        values = np.asarray(action.values, dtype=np.float64)
        axes = np.concatenate((
            values[:3] / self.scale.translation_per_unit_m,
            values[3:6] / self.scale.rotation_per_unit_rad,
        ))
        # No clipping, rounding, automatic frame swap, or silent rate change.
        if not np.isfinite(axes).all() or np.any(np.abs(axes) > 1):
            self._state = "fault_latched"
            raise PermissionError("Candidate exceeds normalized controller-input range")

        # The deployed Robotiq bridge is BINARY: intermediate knuckle positions
        # must never be silently converted to open/close events.
        if np.isclose(values[6], GRIPPER_OPEN_CANDIDATE_RAD, atol=1e-6, rtol=0):
            gripper = "open_preview"
        elif np.isclose(values[6], GRIPPER_CLOSED_CANDIDATE_RAD, atol=1e-6, rtol=0):
            gripper = "close_preview"
        else:
            self._state = "fault_latched"
            raise PermissionError("Intermediate Robotiq target unsupported by observed bridge")

        self._last_sequence = action.sequence
        return {
            "mode": "offline_math_preview_only",
            "preview_state": self._state,
            "sequence": self._last_sequence,
            "illustrative_normalized_axes": axes.tolist(),
            "illustrative_gripper_event": gripper,
            "conversion_semantics_hardware_verified": False,
            "exclusive_controller_authority_verified": False,
            "physical_execution_approved": False,
            "ros_messages_created": 0,
            "robot_commands_sent": 0,
        }

    def disable(self) -> None:
        """Explicit local preview shutdown; does NOT stop a physical robot."""
        self._state = "disabled"
        self._owner = None
        self._heartbeat_ns = None
        self._last_sequence = 0

    def send_to_robot(self, *_args: Any, **_kwargs: Any) -> None:
        raise PermissionError("Policy dispatch is not implemented or authorized")
