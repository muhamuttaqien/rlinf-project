# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Review live pi0.5 SpaceMouse-request predictions before physical rollouts.

Pure, commandless conversion boundary. No ROS imports, robot publisher, action
client or live-execution option. Outputs are NOT approved robot commands.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

AXIS_NAMES = (
    "linear_x", "linear_y", "linear_z",
    "angular_x", "angular_y", "angular_z",
)
GRIPPER_CLOSE = 0.0
GRIPPER_OPEN = 1.0


def preview_request(request: np.ndarray | list[float]) -> dict:
    """Inspect one policy output as a normalized operator request, never an SI twist.

    The SFT seventh channel represents a latched operator button REQUEST:
    0=close, 1=open. Midrange model predictions are ambiguous: HOLD, don't
    guess. Small outside-[0,1] SFT overshoots are quantized for preview.
    """
    x = np.asarray(request, dtype=np.float64)
    if x.shape != (7,) or not np.isfinite(x).all():
        raise ValueError("Expected seven finite request values")
    if np.any(np.abs(x[:6]) > 1.0):
        raise ValueError("Arm axis outside normalized operator range [-1, 1]")
    g = float(x[6])
    if not -0.05 <= g <= 1.05:
        raise ValueError("Gripper prediction too far outside the trained binary range")
    if g <= 0.30:
        intent = "close_request_preview"
    elif g >= 0.70:
        intent = "open_request_preview"
    else:
        intent = "hold_ambiguous_gripper_preview"
    return {
        "normalized_arm_axes": dict(zip(AXIS_NAMES, x[:6].tolist(), strict=True)),
        "gripper_raw": g,
        "gripper_intent": intent,
        "command_message": None,
        "robot_commands_sent": 0,
    }


def summarize_live_shadow(shadow_dir: Path) -> dict:
    shadow_dir = Path(shadow_dir)
    meta = json.loads((shadow_dir / "manifest.json").read_text(encoding="utf-8"))
    if (meta.get("mode") != "live_sensor_shadow_inference_only"
            or meta.get("robot_commands_sent") != 0
            or meta.get("physical_execution_approved") is not False):
        raise ValueError("Expected commandless live shadow manifest")
    path = shadow_dir / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len(rows) != meta.get("samples"):
        raise ValueError("Missing or inconsistent live shadow predictions")
    chunks = []
    durations = []
    gripper_events = {"close_request_preview": 0,
                      "open_request_preview": 0,
                      "hold_ambiguous_gripper_preview": 0}
    for i, row in enumerate(rows):
        if (row.get("sample") != i or row.get("robot_commands_sent") != 0
                or row.get("executed_action") is not None
                or row.get("is_success") is not None):
            raise ValueError("Unexpected shadow record or physical execution")
        values = np.asarray(row.get("predicted_request_chunk_7d"), dtype=np.float64)
        if values.shape != (10, 7):
            raise ValueError("Expected exactly ten 7D requests per prediction")
        for item in values:
            gripper_events[preview_request(item)["gripper_intent"]] += 1
        chunks.append(values)
        duration = row.get("inference_elapsed_s")
        if not isinstance(duration, (float, int)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Missing inference duration")
        durations.append(float(duration))
    arr = np.stack(chunks)
    return {
        "mode": "preexecution_request_preview_only",
        "live_predictions_checked": len(rows),
        "total_predicted_steps": len(rows) * 10,
        "normalized_arm_max_abs": float(np.max(np.abs(arr[..., :6]))),
        "gripper_predicted_min": float(np.min(arr[..., 6])),
        "gripper_predicted_max": float(np.max(arr[..., 6])),
        "gripper_preview_counts": gripper_events,
        "inference_median_s": float(np.median(durations)),
        "inference_max_s": max(durations),
        "controller_request_timeout_s": 0.25,
        "inference_may_outlast_command_timeout": any(d > 0.25 for d in durations),
        "physical_execution_ready": False,
        "robot_commands_sent": 0,
        "remaining_gates": [
            "Verify exclusive controller ownership (SpaceMouse must not compete).",
            "Verify operator enable, independent stop and workcell clearance.",
            "Qualify timed command streaming and stale-observation watchdog.",
            "Qualify gripper action mapping and controller behavior on NUC.",
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shadow-dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(summarize_live_shadow(args.shadow_dir), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
