# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pure, testable request-to-request adapter for FR3 pi0.5 SFT inference.

SFT targets are *time-window averaged normalized operator requests*, NOT
Cartesian SI velocities. The proposed identity axis mapping matches the
six Twist channel names and contains NO invented physical gain. This alone
does not validate the live NUC controller, timing, mounting frame or robot
safety. Live use remains gated by explicit operator mapping verification.

No ROS imports or motor access.
"""

from __future__ import annotations

import math

ARM_AXES = (
    "linear_x", "linear_y", "linear_z",
    "angular_x", "angular_y", "angular_z",
)
GRIPPER_AXIS = "latched_gripper_open_request"


def validate_action_chunk(chunk) -> list[list[float]]:
    """Validate decoded 10x7 original request units; never clip a prediction."""
    if not isinstance(chunk, list) or len(chunk) != 10:
        raise ValueError("Expected exactly ten 7D decoded request actions")
    validated = []
    for i, row in enumerate(chunk):
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError(f"Action {i}: expected seven channels")
        values = []
        for j, value in enumerate(row):
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError(f"Action {i}, channel {j}: nonfinite/non-numeric request")
            value = float(value)
            lo, hi = (-1.0, 1.0) if j < 6 else (-0.05, 1.05)
            if not lo <= value <= hi:
                axis = ARM_AXES[j] if j < 6 else GRIPPER_AXIS
                raise ValueError(
                    f"Action {i} {axis}={value:.6g} outside allowed [{lo}, {hi}]"
                )
            values.append(value)
        validated.append(values)
    return validated


def identity_arm_request(action7) -> tuple[float, ...]:
    """Pass through normalized training request axes WITHOUT extra multiplier.

    This is not a stand-alone permission to execute on physical hardware.
    Callers must enforce live controller/version, exclusivity and safety gates.
    """
    if len(action7) != 7:
        raise ValueError("Exactly seven request channels required")
    row = validate_action_chunk([list(action7) for _ in range(10)])[0]
    return tuple(row[:6])


class GripperRequestFilter:
    """Debounce model-estimated latched 0/1 requests to avoid rapid toggles.

    Commands change only after >=3 consecutive confident 10Hz predictions
    and minimum interval from last goal. This is NOT an acceptance/position
    acknowledgement from the cross-domain Robotiq bridge.
    """

    def __init__(self, *, confirmations: int = 3, min_goal_interval_s: float = 0.8):
        if confirmations < 1 or min_goal_interval_s < 0:
            raise ValueError("Invalid gripper filter thresholds")
        self.confirmations = confirmations
        self.min_goal_interval_s = min_goal_interval_s
        self.reset()

    def reset(self):
        self.latched = None
        self.candidate = None
        self.count = 0
        self.last_goal_at = None

    def step(self, request: float, *, now: float) -> str:
        if not math.isfinite(request) or not -0.05 <= request <= 1.05:
            raise ValueError("Out-of-range gripper prediction")
        candidate = "close" if request <= 0.30 else "open" if request >= 0.70 else None
        if candidate is None or candidate == self.latched:
            self.candidate = None
            self.count = 0
            return "hold"
        if candidate != self.candidate:
            self.candidate = candidate
            self.count = 1
        else:
            self.count += 1
        if self.count < self.confirmations:
            return "hold"
        if self.last_goal_at is not None and now - self.last_goal_at < self.min_goal_interval_s:
            return "hold"
        self.latched = candidate
        self.last_goal_at = now
        self.candidate = None
        self.count = 0
        return candidate
