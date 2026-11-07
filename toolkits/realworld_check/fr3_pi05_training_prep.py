# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0

"""Prepare real FR3 19D OBSERVATIONS ONLY for future pi0.5 training.

No action targets, LeRobot dataset, normalization statistics, model inference,
ROS traffic or robot commands are produced. Existing raw capture is read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from toolkits.realworld_check.fr3_lerobot_preflight import audit_raw_episode
from toolkits.realworld_check.fr3_policy_contract import (
    ACTION_FIELDS, FEATURE_NAMES, candidate_19d_state,
)


def prepare_observations(raw: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Validate the real episode and project every measured state to 19D."""
    raw = Path(raw)
    previous, _ = audit_raw_episode(raw)
    states: list[np.ndarray] = []
    stamps: list[int] = []
    with (raw / "states.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            record = json.loads(line)
            if record.get("frame_index") != index:
                raise ValueError(f"Frame index mismatch at {index}")
            if record.get("training_action") is not None:
                raise ValueError("Real training action unexpectedly present in raw capture")
            if record.get("operator_command_labels") is not None:
                raise ValueError("Raw operator intents are not approved action labels")
            states.append(candidate_19d_state(record["measured_state"]))
            stamp = record.get("recorded_monotonic_ns")
            if type(stamp) is not int or stamp <= 0:
                raise ValueError("Missing monotonic observation timestamp")
            stamps.append(stamp)
    if not states or len(states) != previous["sample_count"]:
        raise ValueError("Raw state count mismatch")
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("Nonincreasing observation timestamps")
    matrix = np.stack(states).astype(np.float32)
    if matrix.shape != (len(stamps), 19) or not np.isfinite(matrix).all():
        raise ValueError("Invalid projected observation matrix")

    report = {
        "mode": "offline_fr3_observation_only",
        "raw_capture": str(raw.resolve()),
        "raw_capture_validated": True,
        "state_shape": list(matrix.shape),
        "state_feature_names": list(FEATURE_NAMES),
        "candidate_state_selection_reviewed_for_training": False,
        "source_frames_calibrated": False,
        "dual_camera_sources": previous["camera_serials"],
        "camera_frames_copied": False,
        "action_dim_proposed": 7,
        "action_fields_unverified": list(ACTION_FIELDS),
        "training_actions_present": False,
        "executed_actions_verified": False,
        "real_action_normalization_available": False,
        "success_labels_available": False,
        "real_sft_ready": False,
        "recap_ready": False,
        "robot_commands_sent": False,
        "next_step": (
            "After lab approval, capture real task demonstrations with independently "
            "verified controller-accepted action targets, timestamps, task outcomes "
            "and camera alignment; compute task-specific norm stats only then."
        ),
    }
    return report, matrix, np.asarray(stamps, dtype=np.int64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="NEW output directory; existing outputs never overwritten")
    args = parser.parse_args()
    report, states, stamps = prepare_observations(args.raw)
    out = args.output.expanduser()
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "candidate_state_19d.npy", states, allow_pickle=False)
    np.save(out / "observation_monotonic_ns.npy", stamps, allow_pickle=False)
    (out / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out.resolve()), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
