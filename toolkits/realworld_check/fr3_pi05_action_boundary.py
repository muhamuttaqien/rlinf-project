# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""FR3 pi0.5 predicted-action quarantine (offline only; no actuation).

A simulation policy emits vectors but the FR3 controller consumes normalized
SpaceMouse intent, not the proposed metric 7D policy action. Do not infer a
physical actuation mapping from array shape, model output, or finite values.

This module intentionally contains NO ROS imports, goal generation, or
controller-command implementation. It cannot authorize robot execution.
"""

from __future__ import annotations

from typing import Any

import numpy as np

ACTION_DIM = 7
POLICY_DOMAIN = "StackCube Panda simulation"
TARGET_ROBOT = "FR3 with Robotiq 2F-85"
CANDIDATE_FIELDS = (
    "delta_tcp_x_m", "delta_tcp_y_m", "delta_tcp_z_m",
    "delta_tcp_rotvec_x_rad", "delta_tcp_rotvec_y_rad",
    "delta_tcp_rotvec_z_rad", "robotiq_target_knuckle_rad",
)
BLOCKERS = (
    "Checkpoint trained for StackCube Panda simulation, not FR3",
    "Candidate 19D FR3 state has no task-specific normalization",
    "No verified correspondence between model action channels and FR3/Robotiq semantics",
    "NUC controller consumes normalized SpaceMouse intent, not metric 7D pose deltas",
    "Missing verified Cartesian frame/tool calibration and action interval",
    "Missing approved exclusive control authority, freshness gate and physical safety sign-off",
)


def quarantine_pi05_prediction(
    prediction: Any, *, checkpoint: str, source_episode: str,
) -> dict[str, Any]:
    """Validate computational output, explicitly prohibit motor command use.

    The values are only inspected and summarized. No command structures or
    remapped controller axes are returned. Both paths must be nonempty to
    preserve provenance in offline logs.
    """
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("Checkpoint provenance required")
    if not isinstance(source_episode, str) or not source_episode.strip():
        raise ValueError("Input episode provenance required")
    try:
        arr = np.asarray(prediction, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Prediction must be numeric") from exc
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != ACTION_DIM:
        raise ValueError("Expected predicted (horizon, 7) channels")
    if not np.isfinite(arr).all():
        raise ValueError("Nonfinite model prediction blocked")
    return {
        "mode": "offline_monitor_only",
        "source_checkpoint": checkpoint,
        "source_episode": source_episode,
        "policy_domain": POLICY_DOMAIN,
        "target_robot": TARGET_ROBOT,
        "raw_prediction_shape": [int(arr.shape[0]), ACTION_DIM],
        "raw_predictions_finite": True,
        "maximum_absolute_raw_channel_value": float(np.max(np.abs(arr))),
        "channel_interpretation": "unknown; candidate physical semantics NOT established",
        "candidate_fields_not_applied": list(CANDIDATE_FIELDS),
        "execution_approved": False,
        "command_messages_created": 0,
        "robot_commands_sent": False,
        "physical_mapping_verified": False,
        "blockers": list(BLOCKERS),
    }


def to_fr3_controller_command(*_args: Any, **_kwargs: Any) -> None:
    """Actuation intentionally unsupported until a separately approved adapter exists."""
    raise PermissionError(
        "Blocked: no FR3/Robotiq policy-action command conversion is authorized. "
        "Do not map simulation vectors to normalized SpaceMouse Twist."
    )
