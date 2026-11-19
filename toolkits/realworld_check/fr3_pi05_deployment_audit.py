# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Read-only SFT/FR3 request-deployment audit before another physical rollout.

Checks dataset lineage, train-only 32D OpenPI quantiles, exact 7D request
channel contract and existing rollout motion evidence. Does NOT approve the
running controller binary, frame convention, timestamp synchronization or
robot safety; those require independent physical validation.

No ROS imports, inference, robot commands, or checkpoint modifications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from toolkits.realworld_check.fr3_pi05_request_adapter import (
    ARM_AXES,
    GripperRequestFilter,
    identity_arm_request,
    validate_action_chunk,
)
from toolkits.realworld_check.fr3_request_openpi_preflight import validate_metadata
from toolkits.realworld_check.fr3_rollout_motion_diagnostic import diagnose


def compare_recorded_requests(
    raw_root: Path, episode: Path, *, expected_episodes: int = 100
) -> dict:
    """Replay saved normalized requests offline; never infer physical approval."""
    import numpy as np

    paths = sorted(raw_root.glob("episode_*/operator_commands.jsonl"))
    if len(paths) != expected_episodes:
        raise ValueError(
            f"Expected {expected_episodes} demonstration recordings, found {len(paths)}"
        )
    raw, intervals, gripper_values = [], [], Counter()
    corpus_hash = hashlib.sha256()
    for path in paths:
        payload = path.read_bytes()
        corpus_hash.update(
            path.parent.name.encode() + b"\0" + hashlib.sha256(payload).digest()
        )
        stamps = []
        for line in payload.splitlines():
            event = json.loads(line)
            if event.get("name") == "operator_twist":
                if (
                    event.get("topic")
                    != "/franka_controller/target_cartesian_velocity_percent"
                    or event.get("message_type") != "geometry_msgs/msg/Twist"
                    or event.get("units")
                    != {
                        "linear": "normalized_controller_input_not_m_per_s",
                        "angular": "normalized_controller_input_not_rad_per_s",
                    }
                ):
                    raise ValueError(
                        "Expected normalized SpaceMouse Twist request units and endpoint"
                    )
                row = [
                    event["message"][part][axis]
                    for part in ("linear", "angular")
                    for axis in ("x", "y", "z")
                ]
                if any(
                    type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 1
                    for v in row
                ):
                    raise ValueError(
                        "Recorded arm requests must be finite and within [-1,1]"
                    )
                raw.append(row)
                stamp = event["received_monotonic_ns"]
                if (
                    type(stamp) is not int
                    or stamp <= 0
                    or (stamps and stamp <= stamps[-1])
                ):
                    raise ValueError(
                        "Expected increasing positive SpaceMouse receipt timestamps"
                    )
                stamps.append(stamp)
            elif event.get("name") == "operator_gripper_request":
                if (
                    event.get("topic") != "/gripper_client/target_gripper_width_percent"
                    or event.get("message_type") != "std_msgs/msg/Float32"
                    or event.get("units") != {"data": "normalized_open_fraction"}
                    or event["message"]["data"] not in (0.0, 1.0)
                ):
                    raise ValueError("Expected binary normalized gripper open fraction")
                gripper_values[str(float(event["message"]["data"]))] += 1
        if len(stamps) < 2:
            raise ValueError("Expected at least two Twist requests per demonstration")
        intervals.extend(np.diff(stamps).tolist())
    policy_path = episode / "policy_command_log.jsonl"
    policy_payload = policy_path.read_bytes()
    events = [json.loads(line) for line in policy_payload.splitlines() if line.strip()]
    if (
        not events
        or len(events) % 10
        or any(
            e.get("event") != "action" or e.get("command_index") != i % 10
            for i, e in enumerate(events)
        )
    ):
        raise ValueError("Expected complete ordered ten-step policy chunks")
    values = [e["requested_7d"] for e in events]
    for i in range(0, len(values), 10):
        validate_action_chunk(values[i : i + 10])
    decoded = np.asarray(values, dtype=float)
    identity = np.asarray([identity_arm_request(row) for row in values])
    sent = np.asarray([e["sent_arm_6d"] for e in events], dtype=float)
    if (
        sent.shape != identity.shape
        or not np.isfinite(sent).all()
        or not np.allclose(sent, identity * 0.15, rtol=0, atol=1e-12)
    ):
        raise ValueError("Saved legacy commands must equal decoded requests times 0.15")
    stamps = np.asarray([e["monotonic_ns"] for e in events], dtype=np.int64)
    deltas = np.diff(stamps)
    if np.any(deltas <= 0):
        raise ValueError("Expected increasing policy event timestamps")
    boundary = np.asarray([e["command_index"] == 0 for e in events[1:]])
    gripper_filter = GripperRequestFilter()
    replayed_gripper = Counter(
        gripper_filter.step(row[6], now=int(stamp) / 1e9)
        for row, stamp in zip(values, stamps)
    )

    def distribution(array):
        return {
            "minimum": array.min(axis=0).tolist(),
            "maximum": array.max(axis=0).tolist(),
            "mean_absolute": np.abs(array).mean(axis=0).tolist(),
            "p95_absolute": np.quantile(np.abs(array), 0.95, axis=0).tolist(),
        }

    def gaps(array):
        array = np.asarray(array, dtype=float) / 1e6
        return {
            "count": len(array),
            "median_ms": float(np.median(array)) if len(array) else None,
            "maximum_ms": float(array.max()) if len(array) else None,
            "above_250ms": int((array > 250).sum()),
        }

    return {
        "mode": "offline_recorded_request_comparison",
        "demonstrations": len(paths),
        "axis_order": list(ARM_AXES),
        "raw_requests": len(raw),
        "raw_request_distribution": distribution(np.asarray(raw)),
        "raw_receipt_gaps": gaps(intervals),
        "raw_effective_hz": len(intervals) * 1e9 / sum(intervals),
        "raw_gripper_requests": dict(gripper_values),
        "policy_actions": len(events),
        "decoded_distribution": distribution(decoded[:, :6]),
        "legacy_sent_distribution": distribution(sent),
        "identity_sent_distribution": distribution(identity),
        "legacy_mapping_max_absolute_error": float(
            np.abs(sent - identity * 0.15).max()
        ),
        "identity_mapping_max_absolute_error": float(
            np.abs(identity - decoded[:, :6]).max()
        ),
        "policy_effective_hz": len(deltas) * 1e9 / int(stamps[-1] - stamps[0]),
        "policy_within_chunk_gaps": gaps(deltas[~boundary]),
        "policy_between_chunk_gaps": gaps(deltas[boundary]),
        "logged_gripper_events": dict(Counter(e["gripper_event"] for e in events)),
        "offline_debounced_gripper_events": dict(replayed_gripper),
        "raw_corpus_sha256": corpus_hash.hexdigest(),
        "policy_log_sha256": hashlib.sha256(policy_payload).hexdigest(),
        "physical_identity_execution_approved": False,
        "remaining_mismatches": [
            "10 Hz window averages cannot reconstruct 100 Hz operator variations",
            "inference gaps intentionally publish neutral",
            "gripper debounce delays latched goals",
            "live binary/configuration, frames/TCP, acceptance and response require physical evidence",
        ],
    }


def audit(
    train: Path, validation: Path, norm_dir: Path, *, episode: Path | None = None
) -> dict:
    # Confirm that normalization matches the exact training provenance.
    meta = validate_metadata(train, validation, norm_dir)
    payload = json.loads((norm_dir / "norm_stats.json").read_text())
    stats = payload.get("norm_stats", payload)
    actions = stats["actions"]
    mean = actions["mean"][:7]
    q01, q99 = actions["q01"][:7], actions["q99"][:7]
    for j, (lo, hi, mu) in enumerate(zip(q01, q99, mean)):
        if not all(math.isfinite(float(x)) for x in (lo, hi, mu)) or hi <= lo:
            raise ValueError(f"Bad action quantile at channel {j}")
    if any(float(lo) < -1.001 or float(hi) > 1.001 for lo, hi in zip(q01[:6], q99[:6])):
        raise ValueError(
            "Training action statistics not normalized SpaceMouse requests"
        )
    report = {
        "mode": "offline_read_only",
        "train_source_episodes": meta["train"]["source_episodes"],
        "validation_source_episodes": meta["validation"]["source_episodes"],
        "train_only_norm_passed": meta["train_only_norm_passed"],
        "normalization_sha256": hashlib.sha256(
            (norm_dir / "norm_stats.json").read_bytes()
        ).hexdigest(),
        "source_episode_disjoint": meta["source_episode_disjoint"],
        "decoded_action_7d_contract": [*ARM_AXES, "latched_gripper_open_request"],
        "model_action_dimension": 32,
        "robot_request_dimension": 7,
        "raw_request_train_mean": [round(float(x), 6) for x in mean],
        "raw_request_train_q01": [round(float(x), 6) for x in q01],
        "raw_request_train_q99": [round(float(x), 6) for x in q99],
        "inference_inverse": "OpenPI Unnormalize on train-only actions, then [:, :7]",
        "existing_live_extra_arm_gain": 0.15,
        "proposed_request_mapping_gain": 1.0,
        "identity_mapping_implemented": True,
        "identity_execution_gate": "hashed physical verification record and explicit per-run operator consent",
        "trained_action_representation": "time-weighted average normalized SpaceMouse input",
        "controller_consumes": "instantaneous normalized Twist request; NUC C++ mapping",
        "controller_binary_version_verified": False,
        "mounting_frame_and_axis_signs_verified": False,
        "action_hold_time_and_latency_verified": False,
        "gripper_bridge_goal_acceptance_verified": False,
        "physical_identity_execution_approved": False,
        "decision": "software_contract_inspected_physical_controller_verification_pending",
    }
    if episode is not None:
        report["existing_episode"] = diagnose(episode / "policy_command_log.jsonl")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train",
        type=Path,
        default=Path("/data1/Muttaqien/fr3_peg_request_lerobot_train_v1"),
    )
    p.add_argument(
        "--validation",
        type=Path,
        default=Path("/data1/Muttaqien/fr3_peg_request_lerobot_validation_v1"),
    )
    p.add_argument(
        "--norm-dir",
        type=Path,
        default=Path("/data1/Muttaqien/fr3_peg_request_norm_v1"),
    )
    p.add_argument("--episode", type=Path, help="Optional existing rollout directory")
    p.add_argument(
        "--raw-root",
        type=Path,
        help="Read-only comparison of all 100 demonstrations with --episode",
    )
    a = p.parse_args()
    if a.raw_root is not None and a.episode is None:
        p.error("--raw-root requires --episode")
    report = audit(a.train, a.validation, a.norm_dir, episode=a.episode)
    if a.raw_root is not None:
        report["recorded_request_comparison"] = compare_recorded_requests(
            a.raw_root, a.episode
        )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
