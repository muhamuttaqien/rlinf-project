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

import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

ARM_AXES = (
    "linear_x",
    "linear_y",
    "linear_z",
    "angular_x",
    "angular_y",
    "angular_z",
)
GRIPPER_AXIS = "latched_gripper_open_request"
MAPPING_CHECKS = tuple(
    f"{axis}_{direction}" for axis in ARM_AXES for direction in ("positive", "negative")
) + (
    "controller_acceptance",
    "tool_frame_alignment",
    "timing_10hz_100hz",
    "neutral_stop",
    "gripper_open_close",
)


def load_mapping_verification(path: str | Path | None) -> dict:
    """Validate an operator's physical qualification record and evidence hashes.

    This checks the attestation's completeness and integrity, not the truth of
    physical measurements or the identity of a currently running controller.
    """
    if not path:
        raise ValueError(
            "identity_verified requires a physical mapping verification record"
        )
    path = Path(path).resolve()
    payload = path.read_bytes()
    record = json.loads(payload)
    expected = {
        "schema": "fr3_pi05_mapping_verification_v1",
        "mapping_mode": "identity_verified",
        "arm_axes": list(ARM_AXES),
        "policy_period_ns": 100_000_000,
        "publish_period_ns": 10_000_000,
    }
    if not isinstance(record, dict) or any(
        record.get(k) != v for k, v in expected.items()
    ):
        raise ValueError(
            "Mapping verification must match the six-axis 10 Hz/100 Hz contract"
        )
    for field in ("operator", "base_frame", "tool_frame"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"Mapping verification requires {field}")
    for field in ("controller_binary_sha256", "controller_config_sha256"):
        if not isinstance(record.get(field), str) or not re.fullmatch(
            r"[0-9a-f]{64}", record[field]
        ):
            raise ValueError(f"Mapping verification requires {field}")
    try:
        verified_at = datetime.fromisoformat(
            record["verified_at_utc"].replace("Z", "+00:00")
        )
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Mapping verification requires verified_at_utc") from exc
    if verified_at.utcoffset() != timedelta(0) or verified_at > datetime.now(
        timezone.utc
    ):
        raise ValueError("Mapping verification must have a past UTC verification time")
    checks = record.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("Mapping verification requires physical checks")
    evidence_hashes = {}
    for name in MAPPING_CHECKS:
        check = checks.get(name)
        if not isinstance(check, dict) or check.get("passed") is not True:
            raise ValueError(f"Physical mapping check not passed: {name}")
        evidence_path, digest = check.get("evidence_path"), check.get("evidence_sha256")
        if not isinstance(evidence_path, str) or not evidence_path.strip():
            raise ValueError(f"Physical mapping check requires evidence: {name}")
        evidence = (path.parent / evidence_path).resolve()
        if evidence not in evidence_hashes:
            hasher = hashlib.sha256()
            size = 0
            with evidence.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    hasher.update(block)
                    size += len(block)
            if not size:
                raise ValueError(f"Physical mapping evidence must be nonempty: {name}")
            evidence_hashes[evidence] = hasher.hexdigest()
        if digest != evidence_hashes[evidence]:
            raise ValueError(f"Physical mapping evidence hash mismatch: {name}")
    return {**record, "record_sha256": hashlib.sha256(payload).hexdigest()}


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
                raise ValueError(
                    f"Action {i}, channel {j}: nonfinite/non-numeric request"
                )
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
        if (
            self.last_goal_at is not None
            and now - self.last_goal_at < self.min_goal_interval_s
        ):
            return "hold"
        self.latched = candidate
        self.last_goal_at = now
        self.candidate = None
        self.count = 0
        return candidate
