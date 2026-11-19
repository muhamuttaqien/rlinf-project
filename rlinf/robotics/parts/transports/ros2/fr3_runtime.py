# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded normalized-request scheduling, independent of ROS and inference."""

from __future__ import annotations

import math

ARM_TOPIC = "/franka_controller/target_cartesian_velocity_percent"
GRIP_TOPIC = "/gripper_client/target_gripper_width_percent"
STATE_PREFIX = "/franka_robot_state_broadcaster"
GRIP_JOINT = "robotiq_85_left_knuckle_joint"
JOINTS = tuple(f"fr3_joint{i}" for i in range(1, 8))


def finite_vector(value: list, size: int) -> list[float]:
    """Require the declared finite vector without changing its units."""
    result = [float(x) for x in value]
    if len(result) != size or not all(math.isfinite(x) for x in result):
        raise ValueError(f"Expected {size} finite scalars")
    return result


def request_vector(value: list) -> list[float]:
    """Validate six unchanged arm requests and a normalized gripper prediction."""
    result = finite_vector(value, 7)
    if any(abs(x) > 1 for x in result[:6]):
        raise ValueError("Arm request outside [-1, 1]; refusing the entire sample")
    if not -0.06 <= result[6] <= 1.06:
        raise ValueError("Gripper prediction outside admitted [-0.06, 1.06]")
    result[6] = min(1.0, max(0.0, result[6]))
    return result


def supervised_duration_s(value: float) -> float:
    """Accept only the reviewed three- and five-second evaluation profiles."""
    if type(value) not in (int, float) or value not in (3, 5):
        raise ValueError(f"Expected a 3 or 5 second supervised window, found {value!r}")
    return float(value)


class RequestWindow:
    """One non-rearmable supervised window with expiring 10 Hz samples."""

    def __init__(self, duration_s: float = 3.0) -> None:
        self.duration_s = supervised_duration_s(duration_s)
        self.started = None
        self.expires = 0.0
        self.last_tick = None
        self.value = [0.0] * 6
        self.reason = None
        self.gripper_candidate = None
        self.gripper_count = 0
        self.last_gripper = None
        self.last_gripper_time = -math.inf

    @property
    def deadline(self) -> float | None:
        """Return the fixed deadline after the first accepted sample."""
        return None if self.started is None else self.started + self.duration_s

    @property
    def deadline_reason(self) -> str:
        """Identify which reviewed supervision window expired."""
        return (
            "three_second_deadline" if self.duration_s == 3 else "five_second_deadline"
        )

    def stop(self, reason: str) -> None:
        """Latch neutral until this process closes."""
        self.reason = self.reason or reason
        self.value = [0.0] * 6

    def submit(self, value: list, now: float, issued: float) -> float | None:
        """Admit one 100 ms sample; return a debounced binary gripper edge."""
        value = request_vector(value)
        if self.reason:
            raise RuntimeError(f"Request window stopped: {self.reason}")
        if not math.isfinite(issued) or not 0 <= now - issued <= 0.3:
            raise ValueError("Request issue time is stale or in the future")
        if self.started is None:
            self.started = now
        if now >= self.deadline:
            self.stop(self.deadline_reason)
            return None
        self.value = value[:6]
        self.expires = min(issued + 0.1, self.deadline)
        edge = 0.0 if value[6] <= 0.3 else 1.0 if value[6] >= 0.7 else None
        if edge != self.gripper_candidate:
            self.gripper_candidate, self.gripper_count = edge, 0
        self.gripper_count += 1
        if (
            edge is not None
            and self.gripper_count >= 3
            and edge != self.last_gripper
            and now - self.last_gripper_time >= 0.8
        ):
            self.last_gripper, self.last_gripper_time = edge, now
            return edge
        return None

    def tick(self, now: float, healthy: bool) -> list[float]:
        """Return requests at 100 Hz, or neutral on any latched failure."""
        if self.started is not None:
            if self.last_tick is not None and now - self.last_tick > 0.05:
                self.stop("command_refresh_gap")
            if now >= self.deadline:
                self.stop(self.deadline_reason)
            if not healthy:
                self.stop("state_or_ownership_lost")
        self.last_tick = now
        return (
            self.value.copy() if not self.reason and now < self.expires else [0.0] * 6
        )
