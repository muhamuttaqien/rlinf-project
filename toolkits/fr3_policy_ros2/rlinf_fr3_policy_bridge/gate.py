# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0.
"""Pure, fail-closed admission gate for isolated ROS 2 shadow messages.

NOT a certified metric-to-controller conversion. This package has no robot
control topic, gripper action, ROS services for arming, or deployment switch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

SCHEMA = "fr3_robotiq_candidate_tcp_delta_v1"
ORIGIN = "synthetic_contract_fixture_only"
FRAME = "fr3_base_candidate_unverified"
ROTATION_FRAME = "base_frame_rotation_vector_candidate_unverified"
CLOCK = "single_host_monotonic_ns"
OWNER = "fr3_shadow_fixture"


@dataclass(frozen=True)
class ShadowProfile:
    # Synthetic test limits only; not workcell/FR3-approved safety limits.
    translation_step_m: float
    rotation_step_rad: float
    max_translation_delta_m: float
    max_rotation_delta_rad: float
    max_observation_age_ns: int
    max_issue_age_ns: int
    max_heartbeat_age_ns: int
    initial_heartbeat_grace_ns: int
    max_interval_ns: int

    def __post_init__(self) -> None:
        floats = (
            self.translation_step_m, self.rotation_step_rad,
            self.max_translation_delta_m, self.max_rotation_delta_rad,
        )
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in floats):
            raise ValueError("Explicit positive, finite synthetic scales/limits required")
        integers = (
            self.max_observation_age_ns, self.max_issue_age_ns,
            self.max_heartbeat_age_ns, self.initial_heartbeat_grace_ns,
            self.max_interval_ns,
        )
        if any(type(x) is not int or x <= 0 for x in integers):
            raise ValueError("Explicit positive integer synthetic age limits required")


SYNTHETIC_PROFILE = ShadowProfile(
    translation_step_m=0.007, rotation_step_rad=0.03,
    max_translation_delta_m=0.005, max_rotation_delta_rad=0.02,
    max_observation_age_ns=200_000_000, max_issue_age_ns=100_000_000,
    max_heartbeat_age_ns=70_000_000,
    initial_heartbeat_grace_ns=10_000_000_000, max_interval_ns=30_000_000,
)


class ShadowRejected(ValueError):
    """Request was rejected and shadow session latched off."""


class ShadowGate:
    """Single synthetic owner, monotonic sequence/clock, heartbeat, fault latch.

    Lifetime is limited to one pre-agreed local shadow session. Re-arming
    requires a NEW process to prevent automatic recovery after a fault.
    """

    def __init__(self, profile: ShadowProfile, *, start_ns: int, lease_ns: int):
        if type(start_ns) is not int or start_ns <= 0:
            raise ValueError("Positive NUC-local monotonic start required")
        if type(lease_ns) is not int or not 0 < lease_ns <= 600_000_000_000:
            raise ValueError("Explicit bounded synthetic lease required")
        self.profile = profile
        self.started_ns = start_ns
        self.expires_ns = start_ns + lease_ns
        self.last_time_ns = start_ns
        self.last_heartbeat_ns: int | None = None
        self.last_sequence = 0
        self.state = "armed_shadow_only"

    def _fault(self, message: str) -> None:
        self.state = "fault_latched"
        raise ShadowRejected(message)

    def _now(self, now_ns: int) -> None:
        if self.state != "armed_shadow_only":
            raise ShadowRejected("Shadow session disabled or fault latched")
        if type(now_ns) is not int or now_ns <= self.last_time_ns:
            self._fault("Non-monotonic shadow receive time")
        self.last_time_ns = now_ns
        if now_ns >= self.expires_ns:
            self._fault("Shadow lease expired")
        if self.last_heartbeat_ns is not None and (
            now_ns - self.last_heartbeat_ns > self.profile.max_heartbeat_age_ns
        ):
            self._fault("Heartbeat expired")

    def heartbeat(self, owner: str, *, now_ns: int) -> None:
        self._now(now_ns)
        if owner != OWNER:
            self._fault("Wrong shadow owner")
        self.last_heartbeat_ns = now_ns

    def check_timeout(self, *, now_ns: int) -> None:
        self._now(now_ns)
        if self.last_heartbeat_ns is None and (
            now_ns - self.started_ns > self.profile.initial_heartbeat_grace_ns
        ):
            self._fault("No independent heartbeat received before startup grace expired")

    def accept(self, record: Any, *, now_ns: int) -> dict[str, Any]:
        self._now(now_ns)
        if self.last_heartbeat_ns is None:
            self._fault("Missing independent shadow heartbeat")
        if not isinstance(record, dict):
            self._fault("Request must be a JSON object")
        fields = {
            "schema", "origin", "owner", "sequence", "clock_domain", "frame",
            "rotation_frame", "source_observation_monotonic_ns",
            "issued_monotonic_ns", "action_interval_ns", "values",
        }
        if set(record) != fields:
            self._fault("Unexpected/missing request fields")
        for key, value in (
            ("schema", SCHEMA), ("origin", ORIGIN), ("owner", OWNER),
            ("clock_domain", CLOCK), ("frame", FRAME),
            ("rotation_frame", ROTATION_FRAME),
        ):
            if record[key] != value:
                self._fault("Unverified request contract: " + key)
        seq = record["sequence"]
        if type(seq) is not int or seq <= self.last_sequence:
            self._fault("Invalid/replayed request sequence")
        observed = record["source_observation_monotonic_ns"]
        issued = record["issued_monotonic_ns"]
        interval = record["action_interval_ns"]
        if (any(type(t) is not int for t in (observed, issued, interval))
                or observed <= 0 or issued < observed or issued > now_ns
                or interval <= 0 or interval > self.profile.max_interval_ns):
            self._fault("Invalid local timestamp/interval")
        if (now_ns - observed > self.profile.max_observation_age_ns
                or now_ns - issued > self.profile.max_issue_age_ns):
            self._fault("Stale observation/command")
        values = record["values"]
        if (not isinstance(values, list) or len(values) != 7
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)):
            self._fault("Seven finite numeric values required")
        position_norm = math.sqrt(sum(float(x) ** 2 for x in values[:3]))
        rotation_norm = math.sqrt(sum(float(x) ** 2 for x in values[3:6]))
        if (position_norm > self.profile.max_translation_delta_m
                or rotation_norm > self.profile.max_rotation_delta_rad):
            self._fault("Synthetic Cartesian bounds exceeded")
        axes = [float(x) / self.profile.translation_step_m for x in values[:3]]
        axes += [float(x) / self.profile.rotation_step_rad for x in values[3:6]]
        if any(not math.isfinite(x) or abs(x) > 1.0 for x in axes):
            self._fault("Normalized controller-axis preview out of range")
        gripper = float(values[6])
        if math.isclose(gripper, 0.0, rel_tol=0, abs_tol=1e-6):
            event = "open_preview"
        elif math.isclose(gripper, 0.7929, rel_tol=0, abs_tol=1e-6):
            event = "close_preview"
        else:
            self._fault("Only observed binary Robotiq endpoints are representable")
        self.last_sequence = seq
        return {
            "sequence": seq,
            "shadow_axes": axes,
            "shadow_gripper_event": event,
            "mode": "shadow_only_no_actuation",
            "metric_to_controller_semantics_verified": False,
            "real_controller_authority": False,
            "robot_commands_sent": 0,
        }

    def disable(self) -> None:
        self.state = "disabled"


def make_synthetic_request(*, sequence: int, observed_ns: int, issued_ns: int,
                           values: list[float] | None = None) -> dict[str, Any]:
    """Construct a clearly synthetic same-host fixture, never a real policy command."""
    return {
        "schema": SCHEMA, "origin": ORIGIN, "owner": OWNER,
        "sequence": sequence, "clock_domain": CLOCK,
        "frame": FRAME, "rotation_frame": ROTATION_FRAME,
        "source_observation_monotonic_ns": observed_ns,
        "issued_monotonic_ns": issued_ns,
        "action_interval_ns": 20_000_000,
        "values": list(values) if values is not None else [0.001, 0, 0, 0, 0, 0, 0],
    }
